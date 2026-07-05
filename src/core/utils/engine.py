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

# Pump ENA — physical pin 27 (PWM5 on controller 34160000). Pin 18 (34150000)
# was unusable: its PWM pads are shared with SPI1, which we keep enabled.
SPLASH_GPIO_PIN: int = int(os.environ.get("SPLASH_GPIO_PIN", "27"))

# L298N motor driver direction pins for the pump.
PUMP_IN1_PIN: int = int(os.environ.get("PUMP_IN1_PIN", "16"))
PUMP_IN2_PIN: int = int(os.environ.get("PUMP_IN2_PIN", "22"))

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
        # Kp is a direct (positional) gain:  new = current + kp * error
        # (no dt scaling).  Ki and Kd remain velocity-form — their
        # contribution is still scaled by dt so integral/derivative behave
        # consistently regardless of tick rate.
        # We still clamp dt for integral/derivative stability when the
        # loop jitters.
        _T = 1.0 / max(1.0, SERVO_LOOP_HZ)
        dt = min(max(now - _pid.last_time, 0.5 * _T), 2.0 * _T)
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
            # kp is positional (no dt scaling); ki/kd are velocity-form
            commanded = max(-90.0, min(90.0,
                current + kp * error + (ki * integral + kd * derivative) * dt))
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
    global _pan_pwm, _tilt_pwm  # noqa: PLW0603
    try:
        stop_servo_loop()
    except Exception:
        pass
    try:
        move_to(0.0, 0.0)
        time.sleep(0.3)
    except Exception:
        pass
    # Stop the software-PWM pump thread (leaves ENA low in its finally block).
    _pump_pwm_stop.set()
    if _pump_pwm_thread is not None:
        _pump_pwm_thread.join(timeout=1.0)
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
            pins = [SERVO_PAN_PIN, SERVO_TILT_PIN, SPLASH_GPIO_PIN, PUMP_IN1_PIN, PUMP_IN2_PIN]
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


_SERVO_LOG_INTERVAL_NS = 1_000_000_000  # throttle per-write servo debug logs to ~1/s
_last_servo_log_ns = 0


def _command_servos(pan_new: float, tilt_new: float, pan_clamped: float, tilt_clamped: float) -> None:
    """Write PWM duty cycles for the given commanded angles."""
    global _last_servo_log_ns  # noqa: PLW0603
    # The servo loop runs at 60 Hz; logging every write floods the log. Only emit
    # the (debug) position lines at most once per second.
    _now_ns = time.monotonic_ns()
    _log_move = logger.isEnabledFor(logging.DEBUG) and (_now_ns - _last_servo_log_ns) >= _SERVO_LOG_INTERVAL_NS
    if _log_move:
        _last_servo_log_ns = _now_ns
    # Read direction/offset from shared-memory mp.Value rather than the app_settings
    # SyncManager proxy: this runs on every PWM write in the 60Hz servo loop, and a
    # DictProxy lookup is a blocking IPC round-trip to the manager process whose latency
    # varies under load — which jitters the loop timing (and thus the servo). mp.Value
    # is local shared memory. Mirrored on change in apps._load_aim_config / api.update_aim_config.
    from ..utils.shared import servo_pan_invert, servo_tilt_invert, vertical_angle_offset  # noqa: PLC0415

    pan_inv = bool(servo_pan_invert.value)
    tilt_inv = bool(servo_tilt_invert.value)
    tilt_offset = vertical_angle_offset.value

    pan_pwm = _get_pan_pwm()
    if pan_pwm is not None:
        try:
            pan_dc = -pan_new if not pan_inv else pan_new
            pan_pwm.ChangeDutyCycle(_angle_to_dc(pan_dc))
            if _log_move:
                logger.debug("Pan servo → %.1f° (target %.1f°)", pan_new, pan_clamped)
        except Exception:
            logger.exception("Failed to move pan servo")

    tilt_pwm = _get_tilt_pwm()
    if tilt_pwm is not None:
        try:
            tilt_out = (tilt_new + tilt_offset) if not tilt_inv else -(tilt_new + tilt_offset)
            tilt_pwm.ChangeDutyCycle(_angle_to_dc(tilt_out))
            if _log_move:
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
        _target.last_pan = pan_angle
        _target.last_tilt = tilt_angle

    return pan_angle, tilt_angle


def release_target() -> None:
    """Signal the servo loop that the lock is gone — it holds its last position."""
    with _target.lock:
        _target.active = False
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

    _PWM_MIN_INTERVAL = 0.05  # throttle PWM writes to 20 Hz max
    _last_pwm_time = 0.0
    next_tick = time.monotonic()

    # Opt-in timing profile (PIKI_SERVO_PROFILE=1): every ~2s log the distribution of
    # the loop's actual tick interval and per-tick work time. A healthy loop shows dt
    # p99 ≈ the nominal period; jitter from contention/IPC shows up as a high p99/max.
    _profile = os.environ.get("PIKI_SERVO_PROFILE") == "1"
    _prof_dt: list[float] = []
    _prof_work: list[float] = []
    _prof_stale: list[float] = []  # measurement age (s) at the moment a PWM command is issued
    _prof_prev = time.monotonic()
    _prof_report = _prof_prev
    _iter_start = _prof_prev

    while not _loop_stop.is_set():
        try:
            if _profile:
                _iter_start = time.monotonic()
                _prof_dt.append(_iter_start - _prof_prev)
                _prof_prev = _iter_start

            # --- Snapshot target state ---
            with _target.lock:
                active = _target.active

            if active and _kalman.is_initialised():
                # Read the filtered position directly — no lookahead projection.
                # The PID loop runs at 60 Hz and already tracks the target; the
                # derivative term handles following a moving target.  Lookahead
                # over-compensates by projecting velocity forward, which causes
                # the predicted aim-point to drift ahead of the measurements
                # when the Kalman velocity estimate is noisy.
                predicted_pan, predicted_tilt = _kalman.predict_lookahead(0.0)

                pan_clamped = max(-90.0, min(90.0, predicted_pan))
                tilt_clamped = max(-90.0, min(90.0, predicted_tilt))

                servo_kalman_pan.value = predicted_pan
                servo_kalman_tilt.value = predicted_tilt

                # Dead zone check + PWM throttle so we don't spam
                # ChangeDutyCycle faster than the servo can track.
                dead_zone = servo_dead_zone.value
                now = time.monotonic()
                if (now - _last_pwm_time >= _PWM_MIN_INTERVAL
                        and (abs(pan_clamped - servo_pan.value) >= dead_zone
                             or abs(tilt_clamped - servo_tilt.value) >= dead_zone)):
                    pan_new, tilt_new = _pid_step(
                        pan_clamped, tilt_clamped,
                        servo_pan.value, servo_tilt.value,
                        servo_pid_kp.value, servo_pid_ki.value, servo_pid_kd.value,
                    )
                    # Round only for the PWM command, not for state tracking.
                    # The PID state must keep float precision so sub-degree
                    # corrections accumulate across ticks instead of being
                    # discarded by rounding every iteration.
                    _command_servos(round(pan_new), round(tilt_new), pan_clamped, tilt_clamped)
                    servo_pan.value = pan_new
                    servo_tilt.value = tilt_new
                    _last_pwm_time = now
                    if _profile and _last_measurement_time > 0.0:
                        # How stale was the last YOLO measurement when we actuated?
                        # This is the control-chain phase lag (servo tick + PWM
                        # throttle + 1-frame feed delay), independent of detection fps.
                        _prof_stale.append(now - _last_measurement_time)

            # else: no active lock — sit idle and hold the last position (don't touch PWM)

        except Exception:
            logger.exception("Servo loop iteration failed")

        if _profile:
            _now = time.monotonic()
            _prof_work.append(_now - _iter_start)
            if _now - _prof_report >= 2.0 and _prof_dt:
                _dt_ms = np.array(_prof_dt) * 1e3
                _wk_ms = np.array(_prof_work) * 1e3
                if _prof_stale:
                    _st_ms = np.array(_prof_stale) * 1e3
                    _stale_str = (" meas_age(ms) p50=%.0f p99=%.0f max=%.0f cmds=%d" % (
                        float(np.percentile(_st_ms, 50)), float(np.percentile(_st_ms, 99)),
                        float(_st_ms.max()), len(_prof_stale)))
                else:
                    _stale_str = " meas_age(ms) n/a (no commands issued)"
                logger.info(
                    "[servo-profile] %d ticks/%.1fs  dt(ms) p50=%.1f p99=%.1f max=%.1f "
                    "nominal=%.1f  work(ms) p50=%.2f p99=%.2f max=%.2f" + _stale_str,
                    len(_prof_dt), _now - _prof_report,
                    float(np.percentile(_dt_ms, 50)), float(np.percentile(_dt_ms, 99)),
                    float(_dt_ms.max()), period * 1e3,
                    float(np.percentile(_wk_ms, 50)), float(np.percentile(_wk_ms, 99)),
                    float(_wk_ms.max()),
                )
                _prof_dt.clear()
                _prof_work.clear()
                _prof_stale.clear()
                _prof_report = _now

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
    """
    # Disable tracking so the servo loop doesn't overwrite this manual move
    # on the very next tick.
    with _target.lock:
        _target.active = False

    pan_clamped = max(-90.0, min(90.0, float(pan_angle)))
    tilt_clamped = max(-90.0, min(90.0, float(tilt_angle)))
    pan_clamped = round(pan_clamped)
    tilt_clamped = round(tilt_clamped)

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
# Pump — L298N motor driver with *software* PWM speed control.
#
#   ENA  →  pin 27 (digital GPIO)  —  bit-banged PWM (~100 Hz) for speed
#   IN1  →  pin 16 (GPIO23)        —  digital HIGH  (forward)
#   IN2  →  pin 22 (GPIO25)        —  digital LOW
#
# Hardware PWM is not used: pin 27 is the ID_SD pad whose PWM5 alternate
# function is never muxed by the device tree (Hobot.GPIO does no pin-muxing of
# its own — it's pure sysfs — so GPIO.PWM exported the channel but no waveform
# reached the pad). Driving ENA as a plain GPIO and toggling it in software
# sidesteps the missing pad-mux while keeping variable speed; a DC pump through
# an L298N doesn't care about the lower frequency / timing jitter.
# ---------------------------------------------------------------------------
_PUMP_SOFT_PWM_HZ = 100          # software-PWM carrier frequency
_pump_initialised = False
_pump_pwm_stop = threading.Event()
_pump_pwm_thread = None


def _init_pump() -> bool:
    global _pump_initialised  # noqa: PLW0603
    if _pump_initialised:
        return True
    if not _init_gpio():
        return False
    try:
        import Hobot.GPIO as GPIO  # noqa: PLC0415

        # May be called from a different thread — ensure the channel mode
        # is set after any cleanup (cleanup resets internal state).
        try:
            GPIO.cleanup([SPLASH_GPIO_PIN, PUMP_IN1_PIN, PUMP_IN2_PIN])
        except Exception:
            pass
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        GPIO.setup(PUMP_IN1_PIN, GPIO.OUT)
        GPIO.output(PUMP_IN1_PIN, GPIO.HIGH)
        GPIO.setup(PUMP_IN2_PIN, GPIO.OUT)
        GPIO.output(PUMP_IN2_PIN, GPIO.LOW)

        # ENA as a plain digital output, held low (pump off) until activated.
        GPIO.setup(SPLASH_GPIO_PIN, GPIO.OUT)
        GPIO.output(SPLASH_GPIO_PIN, GPIO.LOW)

        _pump_initialised = True
        logger.info("Pump initialised: ENA=soft-PWM@%dHz on pin %d (digital GPIO), "
                    "IN1=HIGH on pin %d, IN2=LOW on pin %d",
                    _PUMP_SOFT_PWM_HZ, SPLASH_GPIO_PIN, PUMP_IN1_PIN, PUMP_IN2_PIN)
    except Exception:
        logger.warning("Pump unavailable on pin %d", SPLASH_GPIO_PIN, exc_info=True)
    return _pump_initialised


def _pump_pwm_loop(duty: float, duration_s: float, stop: threading.Event) -> None:
    """Bit-bang ENA at *duty* % for *duration_s* s, then leave it low."""
    import Hobot.GPIO as GPIO  # noqa: PLC0415

    period = 1.0 / _PUMP_SOFT_PWM_HZ
    frac = max(0.0, min(1.0, duty / 100.0))
    on_s = period * frac
    off_s = period - on_s
    deadline = time.perf_counter() + duration_s
    try:
        while not stop.is_set() and time.perf_counter() < deadline:
            if frac >= 1.0:                       # full speed → steady high
                GPIO.output(SPLASH_GPIO_PIN, GPIO.HIGH)
                stop.wait(period)
            elif frac <= 0.0:                     # off → steady low
                GPIO.output(SPLASH_GPIO_PIN, GPIO.LOW)
                stop.wait(period)
            else:
                GPIO.output(SPLASH_GPIO_PIN, GPIO.HIGH)
                if stop.wait(on_s):
                    break
                GPIO.output(SPLASH_GPIO_PIN, GPIO.LOW)
                stop.wait(off_s)
    finally:
        try:
            GPIO.output(SPLASH_GPIO_PIN, GPIO.LOW)
        except Exception:
            logger.exception("Failed to drive pump ENA low")
        logger.info("Pump OFF")


def deactivate_pump() -> None:
    """Immediately stop the pump (cancel any in-flight soft-PWM thread)."""
    global _pump_pwm_thread  # noqa: PLW0603
    _pump_pwm_stop.set()
    if _pump_pwm_thread is not None and _pump_pwm_thread.is_alive():
        _pump_pwm_thread.join(timeout=0.5)
    if _pump_initialised:
        try:
            import Hobot.GPIO as GPIO  # noqa: PLC0415
            GPIO.output(SPLASH_GPIO_PIN, GPIO.LOW)
        except Exception:
            logger.exception("Failed to drive pump ENA low")


def activate_pump(duration_s: float, duty_pct: float = 100.0) -> None:
    """Run the pump at *duty_pct* % for *duration_s* seconds, non-blocking."""
    global _pump_pwm_thread  # noqa: PLW0603
    if not _init_pump():
        return

    duty = max(0.0, min(100.0, duty_pct))

    # Cancel any in-flight run before starting a new one.
    if _pump_pwm_thread is not None and _pump_pwm_thread.is_alive():
        _pump_pwm_stop.set()
        _pump_pwm_thread.join(timeout=1.0)
    _pump_pwm_stop.clear()

    logger.info("Pump ON (duty=%.0f%%, duration=%.1fs, soft-PWM=%dHz)",
                duty, duration_s, _PUMP_SOFT_PWM_HZ)
    _pump_pwm_thread = threading.Thread(
        target=_pump_pwm_loop, args=(duty, duration_s, _pump_pwm_stop),
        daemon=True, name="pump-pwm",
    )
    _pump_pwm_thread.start()
