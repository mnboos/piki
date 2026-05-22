import atexit
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera geometry — linear FOV mapping for the raw ISP output (/image_left_raw).
# ---------------------------------------------------------------------------
SERVO_HFOV: float = float(os.environ.get("SERVO_HFOV", "160.0"))
SERVO_VFOV: float = float(os.environ.get("SERVO_VFOV", "100.0"))

# ---------------------------------------------------------------------------
# GPIO pins for pan and tilt servos (hardware PWM via Hobot.GPIO).
# ---------------------------------------------------------------------------
SERVO_PAN_PIN: int = int(os.environ.get("SERVO_PAN_PIN", "32"))
SERVO_TILT_PIN: int = int(os.environ.get("SERVO_TILT_PIN", "33"))

SPLASH_GPIO_PIN: int = int(os.environ.get("SPLASH_GPIO_PIN", "18"))

# Decoupled servo loop rate (Hz). Higher = smoother, more CPU. 60Hz is roughly
# the max useful rate for a 50Hz hobby servo (the PWM period itself is 20ms).
SERVO_LOOP_HZ: float = float(os.environ.get("SERVO_LOOP_HZ", "60.0"))

_SERVO_FREQ_HZ = 50
_DC_CENTER = 7.5
_DC_RANGE = 5.0


def _angle_to_dc(angle: float) -> float:
    """Convert servo angle (−90…+90°) to PWM duty cycle (2.5…12.5%)."""
    return _DC_CENTER + (angle / 90.0) * _DC_RANGE


# ---------------------------------------------------------------------------
# PID controller state.
# ---------------------------------------------------------------------------
_INTEGRAL_LIMIT: float = 30.0
_RESET_THRESHOLD: float = 15.0


@dataclass
class _PidState:
    integral_pan: float = 0.0
    integral_tilt: float = 0.0
    last_error_pan: float = 0.0
    last_error_tilt: float = 0.0
    last_time: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


_pid = _PidState()


# ---------------------------------------------------------------------------
# Kalman filter — constant-velocity model for ahead-of-time servo aiming.
# ---------------------------------------------------------------------------

class KalmanAimer:
    """Linear Kalman filter for predictive servo aiming.

    Supports two operations:
      - update(): full predict+update cycle when a new measurement arrives.
      - predict_lookahead(): read-only extrapolation so the servo loop can
        project the target's position ahead without mutating filter state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialised = False
        self._x = np.zeros(4)
        self._P = np.eye(4) * 1000.0

    def reset(self) -> None:
        with self._lock:
            self._initialised = False
            self._x = np.zeros(4)
            self._P = np.eye(4) * 1000.0

    def is_initialised(self) -> bool:
        with self._lock:
            return self._initialised

    def _F(self, dt: float) -> np.ndarray:
        return np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def _Q(self, dt: float, process_noise: float) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        return process_noise * np.array([
            [dt4 / 4, 0.0,     dt3 / 2, 0.0    ],
            [0.0,     dt4 / 4, 0.0,     dt3 / 2],
            [dt3 / 2, 0.0,     dt2,     0.0    ],
            [0.0,     dt3 / 2, 0.0,     dt2    ],
        ])

    def update(
        self,
        pan: float,
        tilt: float,
        dt: float,
        process_noise: float,
        measurement_noise: float,
    ) -> tuple[float, float]:
        """Feed one measurement; return the filtered (pan, tilt).

        Does NOT apply lookahead — call predict_lookahead() for the
        servo command angle.
        """
        dt = max(dt, 1e-3)

        with self._lock:
            if not self._initialised:
                self._x = np.array([pan, tilt, 0.0, 0.0])
                self._initialised = True
                return pan, tilt

            F = self._F(dt)
            Q = self._Q(dt, process_noise)

            x_pred = F @ self._x
            P_pred = F @ self._P @ F.T + Q

            H = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ])
            R = np.eye(2) * (measurement_noise ** 2)

            z = np.array([pan, tilt])
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            self._x = x_pred + K @ y
            self._P = (np.eye(4) - K @ H) @ P_pred

            return float(self._x[0]), float(self._x[1])

    def predict_lookahead(self, lookahead_s: float) -> tuple[float, float]:
        """Return where the target will be after lookahead_s seconds.

        Does not mutate state — pure read-only extrapolation.
        """
        with self._lock:
            if not self._initialised:
                return 0.0, 0.0
            if lookahead_s <= 0.0:
                return float(self._x[0]), float(self._x[1])
            F_la = np.array([
                [1.0, 0.0, lookahead_s, 0.0       ],
                [0.0, 1.0, 0.0,         lookahead_s],
                [0.0, 0.0, 1.0,         0.0        ],
                [0.0, 0.0, 0.0,         1.0        ],
            ])
            x_ahead = F_la @ self._x
            return float(x_ahead[0]), float(x_ahead[1])


_kalman = KalmanAimer()
_last_measurement_time: float = 0.0
# Last raw measurement — used for jump detection on the NEXT measurement,
# replacing the buggy comparison against the commanded servo position.
_last_measurement: Optional[tuple[float, float]] = None


# ---------------------------------------------------------------------------
# Target state shared between feed_target() (called from on_done) and the
# servo loop. Updated under _target_lock.
# ---------------------------------------------------------------------------
@dataclass
class _TargetState:
    active: bool = False         # True when a lock is being fed
    home_pending: bool = False   # True when a release was requested
    last_pan: float = 0.0
    last_tilt: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


_target = _TargetState()


def _pid_step(
    target_pan: float,
    target_tilt: float,
    current_pan: float,
    current_tilt: float,
    kp: float,
    ki: float,
    kd: float,
) -> tuple[float, float]:
    """Advance the PID controller by one step and return the new commanded angles."""
    with _pid.lock:
        now = time.monotonic()
        dt = max(now - _pid.last_time, 1e-3)
        _pid.last_time = now

        new_pan = current_pan
        new_tilt = current_tilt
        for axis, (target, current, i_attr, le_attr) in enumerate([
            (target_pan, current_pan, "integral_pan", "last_error_pan"),
            (target_tilt, current_tilt, "integral_tilt", "last_error_tilt"),
        ]):
            error = target - current
            if abs(error) > _RESET_THRESHOLD:
                setattr(_pid, i_attr, 0.0)
            integral = getattr(_pid, i_attr) + error * dt
            integral = max(-_INTEGRAL_LIMIT, min(_INTEGRAL_LIMIT, integral))
            setattr(_pid, i_attr, integral)
            last_error = getattr(_pid, le_attr)
            derivative = (error - last_error) / dt
            setattr(_pid, le_attr, error)
            output = kp * error + ki * integral + kd * derivative
            commanded = max(-90.0, min(90.0, current + output))
            if axis == 0:
                new_pan = commanded
            else:
                new_tilt = commanded

    return new_pan, new_tilt


def _pid_reset() -> None:
    """Reset PID accumulators (call after a lock change / servo home)."""
    with _pid.lock:
        _pid.integral_pan = 0.0
        _pid.integral_tilt = 0.0
        _pid.last_error_pan = 0.0
        _pid.last_error_tilt = 0.0
        _pid.last_time = time.monotonic()


# ---------------------------------------------------------------------------
# Servos — initialised lazily.
# ---------------------------------------------------------------------------
_pan_pwm = None
_tilt_pwm = None
_gpio_initialised = False


@atexit.register
def _cleanup_gpio() -> None:
    global _pan_pwm, _tilt_pwm, _splash_pin  # noqa: PLW0603
    try:
        stop_servo_loop()
    except Exception:
        pass
    try:
        move_to(0.0, 0.0)
        time.sleep(0.3)
    except Exception:
        pass
    for attr, pwm in (("_pan_pwm", _pan_pwm), ("_tilt_pwm", _tilt_pwm)):
        if pwm is not None:
            try:
                pwm.stop()
            except Exception:
                pass
    _pan_pwm = None
    _tilt_pwm = None
    if _gpio_initialised:
        try:
            import Hobot.GPIO as GPIO  # noqa: PLC0415
            pins = [SERVO_PAN_PIN, SERVO_TILT_PIN]
            if _splash_pin is not None:
                GPIO.output(SPLASH_GPIO_PIN, GPIO.LOW)
                pins.append(SPLASH_GPIO_PIN)
            GPIO.cleanup(pins)
        except Exception:
            pass


def _init_gpio() -> bool:
    global _gpio_initialised  # noqa: PLW0603
    if _gpio_initialised:
        return True
    try:
        import Hobot.GPIO as GPIO  # noqa: PLC0415
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        try:
            GPIO.cleanup([SERVO_PAN_PIN, SERVO_TILT_PIN])
        except Exception:
            pass
        _gpio_initialised = True
        return True
    except Exception:
        logger.warning("Hobot.GPIO unavailable — servos will be simulated", exc_info=True)
        return False


def _get_pan_pwm():
    global _pan_pwm  # noqa: PLW0603
    if _pan_pwm is not None:
        return _pan_pwm
    if not _init_gpio():
        return None
    try:
        import Hobot.GPIO as GPIO  # noqa: PLC0415
        _pan_pwm = GPIO.PWM(SERVO_PAN_PIN, _SERVO_FREQ_HZ)
        _pan_pwm.ChangeDutyCycle(_DC_CENTER)
        _pan_pwm.start(_DC_CENTER)
        logger.info("Pan servo initialised on physical pin %d (hardware PWM)", SERVO_PAN_PIN)
    except Exception:
        logger.warning("Pan servo unavailable on pin %d", SERVO_PAN_PIN, exc_info=True)
        _pan_pwm = None
    return _pan_pwm


def _get_tilt_pwm():
    global _tilt_pwm  # noqa: PLW0603
    if _tilt_pwm is not None:
        return _tilt_pwm
    if not _init_gpio():
        return None
    try:
        import Hobot.GPIO as GPIO  # noqa: PLC0415
        _tilt_pwm = GPIO.PWM(SERVO_TILT_PIN, _SERVO_FREQ_HZ)
        _tilt_pwm.ChangeDutyCycle(_DC_CENTER)
        _tilt_pwm.start(_DC_CENTER)
        logger.info("Tilt servo initialised on physical pin %d (hardware PWM)", SERVO_TILT_PIN)
    except Exception:
        logger.warning("Tilt servo unavailable on pin %d", SERVO_TILT_PIN, exc_info=True)
        _tilt_pwm = None
    return _tilt_pwm


def bbox_to_angles(bbox_normalized: list[float]) -> tuple[float, float]:
    """Convert a normalised bounding box centre to pan/tilt angles (degrees)."""
    ymin, xmin, ymax, xmax = bbox_normalized
    cx_n = (xmin + xmax) / 2.0
    cy_n = (ymin + ymax) / 2.0
    pan_angle = (cx_n - 0.5) * SERVO_HFOV
    tilt_angle = (cy_n - 0.5) * SERVO_VFOV
    return pan_angle, tilt_angle


def _command_servos(pan_new: float, tilt_new: float, pan_clamped: float, tilt_clamped: float) -> None:
    """Write PWM duty cycles for the given commanded angles."""
    from ..utils.shared import app_settings, vertical_angle_offset  # noqa: PLC0415

    pan_inv = app_settings.aim_settings.pan_invert
    tilt_inv = app_settings.aim_settings.tilt_invert
    tilt_offset = vertical_angle_offset.value

    pan_pwm = _get_pan_pwm()
    if pan_pwm is not None:
        try:
            pan_dc = -pan_new if not pan_inv else pan_new
            pan_pwm.ChangeDutyCycle(_angle_to_dc(pan_dc))
            logger.debug("Pan servo → %.1f° (target %.1f°)", pan_new, pan_clamped)
        except Exception:
            logger.exception("Failed to move pan servo")

    tilt_pwm = _get_tilt_pwm()
    if tilt_pwm is not None:
        try:
            tilt_out = (tilt_new + tilt_offset) if not tilt_inv else -(tilt_new + tilt_offset)
            tilt_pwm.ChangeDutyCycle(_angle_to_dc(tilt_out))
            logger.debug("Tilt servo → %.1f° (target %.1f° offset=%.1f°)", tilt_out, tilt_clamped, tilt_offset)
        except Exception:
            logger.exception("Failed to move tilt servo")


def feed_target(
    bbox_normalized: list[float],
    aim_center: "tuple[float, float] | None" = None,
) -> tuple[float, float]:
    """Feed one YOLO measurement into the Kalman filter.

    Called from on_done() in stream.py when a new inference completes.
    Does NOT command the servo directly — the servo loop reads the Kalman
    state at its own rate and issues PWM updates.

    Returns the raw (pan, tilt) angles for logging.
    """
    global _last_measurement_time, _last_measurement  # noqa: PLW0603

    from . import exclusion as _exclusion  # noqa: PLC0415

    if aim_center is not None:
        if _exclusion.point_inside_any(aim_center):
            logger.debug("Aim suppressed: centroid (%.3f, %.3f) inside exclusion zone", *aim_center)
            return 0.0, 0.0
        cx_n, cy_n = aim_center
        pan_angle = (cx_n - 0.5) * SERVO_HFOV
        tilt_angle = (cy_n - 0.5) * SERVO_VFOV
    else:
        if _exclusion.bbox_centroid_inside_any(bbox_normalized):
            logger.debug("Aim suppressed: bbox %s inside exclusion zone", bbox_normalized)
            return 0.0, 0.0
        pan_angle, tilt_angle = bbox_to_angles(bbox_normalized)

    from ..utils.shared import (  # noqa: PLC0415
        servo_kalman_meas_noise,
        servo_kalman_process_noise,
    )

    now = time.monotonic()
    dt = now - _last_measurement_time if _last_measurement_time > 0.0 else 0.033
    _last_measurement_time = now

    # Jump detection: compare against previous MEASUREMENT, not the commanded
    # servo position. This fixes the bug where PID lag could trip a spurious
    # Kalman reset during a sustained chase.
    do_reset = False
    if _last_measurement is not None:
        prev_pan, prev_tilt = _last_measurement
        if abs(pan_angle - prev_pan) > _RESET_THRESHOLD or abs(tilt_angle - prev_tilt) > _RESET_THRESHOLD:
            do_reset = True

    if do_reset:
        # Reset BEFORE feeding the new measurement so the filter bootstraps
        # cleanly from the new position with zero velocity, rather than the
        # original bug of absorbing the jump and then resetting.
        _kalman.reset()
        _pid_reset()
        logger.info("Kalman reset on measurement jump: prev=%s new=(%.1f°, %.1f°)",
                    _last_measurement, pan_angle, tilt_angle)

    _last_measurement = (pan_angle, tilt_angle)

    _kalman.update(
        pan_angle, tilt_angle, dt,
        process_noise=servo_kalman_process_noise.value,
        measurement_noise=servo_kalman_meas_noise.value,
    )

    with _target.lock:
        _target.active = True
        _target.home_pending = False
        _target.last_pan = pan_angle
        _target.last_tilt = tilt_angle

    return pan_angle, tilt_angle


def release_target() -> None:
    """Signal the servo loop that the lock is gone — it will home the servos."""
    with _target.lock:
        _target.active = False
        _target.home_pending = True
    _kalman.reset()
    _pid_reset()
    global _last_measurement, _last_measurement_time  # noqa: PLW0603
    _last_measurement = None
    _last_measurement_time = 0.0


# ---------------------------------------------------------------------------
# Decoupled servo command loop.
#
# Runs at SERVO_LOOP_HZ in a daemon thread, independent of YOLO inference
# completion rate. Reads the Kalman state, applies lookahead, and commands
# the servo. This is what makes the servo "near real-time" — it ticks at
# 60Hz regardless of whether inference is at 5Hz or 20Hz.
# ---------------------------------------------------------------------------
_loop_thread: Optional[threading.Thread] = None
_loop_stop = threading.Event()


def _servo_loop() -> None:
    period = 1.0 / max(1.0, SERVO_LOOP_HZ)
    logger.info("Servo loop started at %.0fHz (period=%.1fms)", SERVO_LOOP_HZ, period * 1000)

    from ..utils.shared import (  # noqa: PLC0415
        servo_dead_zone,
        servo_kalman_lookahead_ms,
        servo_kalman_pan,
        servo_kalman_tilt,
        servo_pan,
        servo_pid_kd,
        servo_pid_ki,
        servo_pid_kp,
        servo_tilt,
    )

    next_tick = time.monotonic()

    while not _loop_stop.is_set():
        try:
            # --- Snapshot target state ---
            with _target.lock:
                active = _target.active
                home_pending = _target.home_pending

            if home_pending:
                # Lock released — drive servos to neutral, bypassing dead zone.
                pan_new, tilt_new = _pid_step(
                    0.0, 0.0,
                    servo_pan.value, servo_tilt.value,
                    servo_pid_kp.value, servo_pid_ki.value, servo_pid_kd.value,
                )
                _command_servos(pan_new, tilt_new, 0.0, 0.0)
                servo_pan.value = pan_new
                servo_tilt.value = tilt_new
                servo_kalman_pan.value = 0.0
                servo_kalman_tilt.value = 0.0
                # Consume home_pending only once the servos are near neutral,
                # so the PID has enough ticks to actually get there.
                if abs(pan_new) < 0.5 and abs(tilt_new) < 0.5:
                    with _target.lock:
                        _target.home_pending = False

            elif active and _kalman.is_initialised():
                # Extrapolate by a FIXED lookahead only (servo-lag compensation),
                # read-only from the last measurement-updated state.  Do NOT scale
                # the horizon by time-since-measurement: motion-gated inference stops
                # feeding the filter when the scene goes still, so an elapsed-scaled
                # horizon would keep marching the aim point in the last-known velocity
                # direction every tick — the stepwise "wander".  With a fixed horizon a
                # frozen state gives a constant prediction (and v≈0 ⇒ it holds still).
                # State is advanced only by feed_target()/update(), once per measurement.
                lookahead_s = servo_kalman_lookahead_ms.value / 1000.0
                predicted_pan, predicted_tilt = _kalman.predict_lookahead(lookahead_s)

                pan_clamped = max(-90.0, min(90.0, predicted_pan))
                tilt_clamped = max(-90.0, min(90.0, predicted_tilt))

                servo_kalman_pan.value = predicted_pan
                servo_kalman_tilt.value = predicted_tilt

                # Dead zone check on the loop's commanded position so we don't
                # spam ChangeDutyCycle for sub-degree moves.
                dead_zone = servo_dead_zone.value
                if (abs(pan_clamped - servo_pan.value) >= dead_zone
                        or abs(tilt_clamped - servo_tilt.value) >= dead_zone):
                    pan_new, tilt_new = _pid_step(
                        pan_clamped, tilt_clamped,
                        servo_pan.value, servo_tilt.value,
                        servo_pid_kp.value, servo_pid_ki.value, servo_pid_kd.value,
                    )
                    _command_servos(pan_new, tilt_new, pan_clamped, tilt_clamped)
                    servo_pan.value = pan_new
                    servo_tilt.value = tilt_new

            # else: no active lock and no home request — sit idle, don't touch PWM

        except Exception:
            logger.exception("Servo loop iteration failed")

        # Sleep until next tick.
        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            _loop_stop.wait(timeout=sleep_for)
        else:
            # Loop is running behind schedule — reset the clock so we don't
            # spiral.
            next_tick = time.monotonic()

    logger.info("Servo loop stopped.")


def start_servo_loop() -> None:
    """Start the decoupled servo command thread (idempotent)."""
    global _loop_thread  # noqa: PLW0603
    if _loop_thread is not None and _loop_thread.is_alive():
        return
    _loop_stop.clear()
    _loop_thread = threading.Thread(target=_servo_loop, daemon=True, name="piki-servo")
    _loop_thread.start()


def stop_servo_loop() -> None:
    """Signal the servo loop to exit and wait briefly for it."""
    global _loop_thread  # noqa: PLW0603
    if _loop_thread is None:
        return
    _loop_stop.set()
    _loop_thread.join(timeout=1.0)
    _loop_thread = None


# ---------------------------------------------------------------------------
# Backwards-compatible alias. Some call sites (replay, test harness) may still
# import aim_at; route it through feed_target.
# ---------------------------------------------------------------------------
def aim_at(
    bbox_normalized: list[float],
    aim_center: "tuple[float, float] | None" = None,
    depth_map: "Optional[object]" = None,
) -> tuple[float, float]:
    """Legacy entry point — feeds a measurement into the servo loop's Kalman.

    The servo will move on the next loop tick; this no longer blocks on PWM.
    """
    return feed_target(bbox_normalized=bbox_normalized, aim_center=aim_center)


def move_to(pan_angle: float, tilt_angle: float) -> tuple[float, float]:
    """Directly command both servos to the given angles (manual / debug mode).

    Bypasses Kalman / PID / dead zone — used by the debug panel and cleanup.
    For normal lock-release homing the servo loop PID-steps to neutral.
    """
    pan_clamped = max(-90.0, min(90.0, float(pan_angle)))
    tilt_clamped = max(-90.0, min(90.0, float(tilt_angle)))

    logger.info("Manual move → pan=%.1f° tilt=%.1f°", pan_clamped, tilt_clamped)

    from ..utils.shared import app_settings, vertical_angle_offset  # noqa: PLC0415
    pan_inv = app_settings.aim_settings.pan_invert
    tilt_inv = app_settings.aim_settings.tilt_invert
    tilt_offset = vertical_angle_offset.value

    pan_pwm = _get_pan_pwm()
    if pan_pwm is not None:
        try:
            pan_dc = -pan_clamped if not pan_inv else pan_clamped
            pan_pwm.ChangeDutyCycle(_angle_to_dc(pan_dc))
        except Exception:
            logger.exception("Failed to move pan servo")

    tilt_pwm = _get_tilt_pwm()
    if tilt_pwm is not None:
        try:
            tilt_out = (tilt_clamped + tilt_offset) if not tilt_inv else -(tilt_clamped + tilt_offset)
            tilt_pwm.ChangeDutyCycle(_angle_to_dc(tilt_out))
        except Exception:
            logger.exception("Failed to move tilt servo")

    from ..utils.shared import servo_pan, servo_tilt  # noqa: PLC0415
    servo_pan.value = pan_clamped
    servo_tilt.value = tilt_clamped

    return pan_clamped, tilt_clamped


# ---------------------------------------------------------------------------
# Splash relay — digital GPIO output for a solenoid / water valve.
# ---------------------------------------------------------------------------
_splash_pin = None


def _init_splash_gpio() -> bool:
    global _splash_pin  # noqa: PLW0603
    if _splash_pin is not None:
        return True
    if not _init_gpio():
        return False
    try:
        import Hobot.GPIO as GPIO  # noqa: PLC0415
        GPIO.setup(SPLASH_GPIO_PIN, GPIO.OUT)
        GPIO.output(SPLASH_GPIO_PIN, GPIO.LOW)
        _splash_pin = True
        logger.info("Splash relay initialised on physical pin %d", SPLASH_GPIO_PIN)
    except Exception:
        logger.warning("Splash relay unavailable on pin %d", SPLASH_GPIO_PIN, exc_info=True)
        _splash_pin = None
    return _splash_pin is not None


def activate_splash(duration_s: float) -> None:
    """Activate the splash relay for *duration_s* seconds, non-blocking."""
    if not _init_splash_gpio():
        return
    import Hobot.GPIO as GPIO  # noqa: PLC0415

    GPIO.output(SPLASH_GPIO_PIN, GPIO.HIGH)
    logger.info("Splash relay ON (duration=%.1fs)", duration_s)

    def _deactivate() -> None:
        time.sleep(duration_s)
        try:
            GPIO.output(SPLASH_GPIO_PIN, GPIO.LOW)
            logger.info("Splash relay OFF")
        except Exception:
            logger.exception("Failed to deactivate splash relay")

    t = threading.Thread(target=_deactivate, daemon=True, name="splash-timer")
    t.start()
