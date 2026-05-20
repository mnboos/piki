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
#
# The ISP already undistorts the lens, so the output is rectilinear (straight
# lines stay straight).  We use a simple linear mapping from normalised frame
# position to angle rather than calibrated pinhole intrinsics:
#
#   pan_angle  = (cx_normalised − 0.5) × SERVO_HFOV
#   tilt_angle = (cy_normalised − 0.5) × SERVO_VFOV
#
# Set SERVO_HFOV / SERVO_VFOV to match your camera's actual field of view.
# The SC230AI + ISP on the RDK X5 outputs roughly 160° × 100° after undistortion.
# Increase toward 180° / 120° to use more of the servo's physical range.
# ---------------------------------------------------------------------------
SERVO_HFOV: float = float(os.environ.get("SERVO_HFOV", "160.0"))  # horizontal FOV in degrees
SERVO_VFOV: float = float(os.environ.get("SERVO_VFOV", "100.0"))  # vertical FOV in degrees

# ---------------------------------------------------------------------------
# GPIO pins for pan and tilt servos (hardware PWM via Hobot.GPIO).
#
# RDK X5 hardware PWM pins (physical / BOARD numbering):
#   Physical 32  →  PWM6  →  pan  (default)
#   Physical 33  →  PWM7  →  tilt (default)
#
# These require the dtoverlay_pwm3 overlay, enabled via /boot/config.txt.
# Override with SERVO_PAN_PIN / SERVO_TILT_PIN env vars (physical pin numbers).
# ---------------------------------------------------------------------------
SERVO_PAN_PIN: int = int(os.environ.get("SERVO_PAN_PIN", "32"))
SERVO_TILT_PIN: int = int(os.environ.get("SERVO_TILT_PIN", "33"))

# GPIO pin for the splash relay/solenoid (digital output, not PWM).
# Override with SPLASH_GPIO_PIN env var (physical pin number).
SPLASH_GPIO_PIN: int = int(os.environ.get("SPLASH_GPIO_PIN", "36"))

# Standard 50 Hz servo PWM: 1.5 ms centre pulse → 7.5% duty cycle.
# Mapping: angle [-90°, +90°] → duty cycle [2.5%, 12.5%]
_SERVO_FREQ_HZ = 50
_DC_CENTER = 7.5
_DC_RANGE = 5.0  # ±5% spans ±90°


def _angle_to_dc(angle: float) -> float:
    """Convert servo angle (−90…+90°) to PWM duty cycle (2.5…12.5%)."""
    return _DC_CENTER + (angle / 90.0) * _DC_RANGE


# ---------------------------------------------------------------------------
# PID controller state — one set of accumulators for pan and tilt.
# Both YOLO and tracker callers run in the same process, so a threading.Lock
# is sufficient (no multiprocessing synchronisation needed here).
# ---------------------------------------------------------------------------
_INTEGRAL_LIMIT: float = 30.0   # °  — clamps per-axis integral wind-up
_RESET_THRESHOLD: float = 15.0  # °  — error jump magnitude that resets integral


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
#
# State: x = [pan, tilt, v_pan, v_tilt]  (degrees / degrees·s⁻¹)
# Observation: z = [pan, tilt]
#
# The filter is updated on every aim_at() call.  After the update step the
# state is projected forward by `lookahead_s` seconds so the servo is
# commanded to where the object *will be* rather than where it *was*.
#
# Tuning parameters (all live-adjustable via shared.py mp.Value):
#   servo_kalman_process_noise — governs how fast velocity may change (deg/s²)
#   servo_kalman_meas_noise    — trust in each position measurement (deg)
#   servo_kalman_lookahead_ms  — servo lag to compensate for (ms)
# ---------------------------------------------------------------------------

class KalmanAimer:
    """Linear Kalman filter for predictive servo aiming."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialised = False
        # State vector [pan, tilt, v_pan, v_tilt]; covariance matrix P (4×4)
        self._x = np.zeros(4)
        self._P = np.eye(4) * 1000.0  # large initial uncertainty

    def reset(self) -> None:
        with self._lock:
            self._initialised = False
            self._x = np.zeros(4)
            self._P = np.eye(4) * 1000.0

    def update(
        self,
        pan: float,
        tilt: float,
        dt: float,
        process_noise: float,
        measurement_noise: float,
        lookahead_s: float,
    ) -> tuple[float, float]:
        """Feed one measurement and return the lookahead-predicted (pan, tilt).

        On the very first call the filter is bootstrapped from the measurement
        with zero velocity so the output equals the input.
        """
        dt = max(dt, 1e-3)

        with self._lock:
            if not self._initialised:
                self._x = np.array([pan, tilt, 0.0, 0.0])
                self._initialised = True
                return pan, tilt

            # --- Predict ---
            F = np.array([
                [1.0, 0.0, dt,  0.0],
                [0.0, 1.0, 0.0, dt ],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            # Continuous white-noise acceleration model discretised to dt
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt3 * dt
            Q = process_noise * np.array([
                [dt4 / 4, 0.0,     dt3 / 2, 0.0    ],
                [0.0,     dt4 / 4, 0.0,     dt3 / 2],
                [dt3 / 2, 0.0,     dt2,     0.0    ],
                [0.0,     dt3 / 2, 0.0,     dt2    ],
            ])

            x_pred = F @ self._x
            P_pred = F @ self._P @ F.T + Q

            # --- Update ---
            H = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ])
            R = np.eye(2) * (measurement_noise ** 2)

            z = np.array([pan, tilt])
            y = z - H @ x_pred                              # innovation
            S = H @ P_pred @ H.T + R                       # innovation covariance
            K = P_pred @ H.T @ np.linalg.inv(S)            # Kalman gain
            self._x = x_pred + K @ y
            self._P = (np.eye(4) - K @ H) @ P_pred

            # --- Lookahead prediction ---
            if lookahead_s > 0.0:
                F_la = np.array([
                    [1.0, 0.0, lookahead_s, 0.0       ],
                    [0.0, 1.0, 0.0,         lookahead_s],
                    [0.0, 0.0, 1.0,         0.0        ],
                    [0.0, 0.0, 0.0,         1.0        ],
                ])
                x_ahead = F_la @ self._x
            else:
                x_ahead = self._x

            return float(x_ahead[0]), float(x_ahead[1])


_kalman = KalmanAimer()
_last_aim_time: float = 0.0


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
        dt = max(now - _pid.last_time, 1e-3)  # floor at 1 ms to avoid division by zero
        _pid.last_time = now

        new_pan = current_pan
        new_tilt = current_tilt
        for axis, (target, current, i_attr, le_attr) in enumerate([
            (target_pan, current_pan, "integral_pan", "last_error_pan"),
            (target_tilt, current_tilt, "integral_tilt", "last_error_tilt"),
        ]):
            error = target - current
            # Large jump → new detection target; reset integral to avoid wind-up carry-over.
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


# ---------------------------------------------------------------------------
# Servos — initialised lazily so that import failures don't crash the app.
# Hobot.GPIO is the correct GPIO library for the RDK X5 (RPi.GPIO-compatible).
# ---------------------------------------------------------------------------
_pan_pwm = None
_tilt_pwm = None
_gpio_initialised = False


@atexit.register
def _cleanup_gpio() -> None:
    global _pan_pwm, _tilt_pwm, _splash_pin  # noqa: PLW0603
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
        # Best-effort release of stale hardware state from a previous (crashed/restarted)
        # process so that GPIO.PWM() does not raise "This channel is in use".
        # Hobot.GPIO raises KeyError if the pin was never setup(), so we ignore errors.
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
        _pan_pwm.ChangeDutyCycle(_DC_CENTER)  # pre-populate sysfs duty so start() enables the channel
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
        _tilt_pwm.ChangeDutyCycle(_DC_CENTER)  # pre-populate sysfs duty so start() enables the channel
        _tilt_pwm.start(_DC_CENTER)
        logger.info("Tilt servo initialised on physical pin %d (hardware PWM)", SERVO_TILT_PIN)
    except Exception:
        logger.warning("Tilt servo unavailable on pin %d", SERVO_TILT_PIN, exc_info=True)
        _tilt_pwm = None
    return _tilt_pwm


def bbox_to_angles(bbox_normalized: list[float]) -> tuple[float, float]:
    """Convert a normalised bounding box centre to pan/tilt angles (degrees).

    Uses a linear FOV mapping against the raw ISP output (/image_left_raw).
    The ISP already undistorts the lens so straight lines stay straight; a
    linear model is both simpler and more appropriate than a calibrated pinhole.

    Args:
        bbox_normalized: ``[ymin, xmin, ymax, xmax]`` normalised to ``[0, 1]``.

    Returns:
        ``(pan_angle, tilt_angle)`` in degrees.  Positive pan = right of centre,
        positive tilt = below centre.
    """
    ymin, xmin, ymax, xmax = bbox_normalized
    cx_n = (xmin + xmax) / 2.0
    cy_n = (ymin + ymax) / 2.0
    pan_angle = (cx_n - 0.5) * SERVO_HFOV
    tilt_angle = (cy_n - 0.5) * SERVO_VFOV
    return pan_angle, tilt_angle


def aim_at(
    bbox_normalized: list[float],
    aim_center: "tuple[float, float] | None" = None,
    depth_map: "Optional[object]" = None,
) -> tuple[float, float]:
    """Aim both pan and tilt servos at a detected object.

    Args:
        bbox_normalized: Detection bounding box as ``[ymin, xmin, ymax, xmax]``
                         normalised to ``[0, 1]`` relative to the actual frame.
        aim_center:      Optional ``(cx_n, cy_n)`` override for the aim point,
                         both normalised to ``[0, 1]``.  When provided (e.g. a
                         foreground-mask centroid), this is used instead of the
                         geometric bbox centre to reduce jitter caused by bbox
                         edge noise.
        depth_map:       Ignored — kept for call-site compatibility.

    Returns:
        ``(pan_angle, tilt_angle)`` in degrees (positive = right / down).
    """
    global _last_aim_time  # noqa: PLW0603

    if aim_center is not None:
        cx_n, cy_n = aim_center
        pan_angle = (cx_n - 0.5) * SERVO_HFOV
        tilt_angle = (cy_n - 0.5) * SERVO_VFOV
        logger.debug("Aim override: centroid (%.3f, %.3f) → pan=%.1f° tilt=%.1f°", cx_n, cy_n, pan_angle, tilt_angle)
    else:
        pan_angle, tilt_angle = bbox_to_angles(bbox_normalized)
    logger.debug("Target at pan=%.1f° tilt=%.1f°", pan_angle, tilt_angle)

    from ..utils.shared import (  # noqa: PLC0415
        app_settings,
        servo_dead_zone,
        servo_kalman_lookahead_ms,
        servo_kalman_meas_noise,
        servo_kalman_pan,
        servo_kalman_process_noise,
        servo_kalman_tilt,
        servo_pan,
        servo_pid_kd,
        servo_pid_ki,
        servo_pid_kp,
        servo_tilt,
    )

    # -----------------------------------------------------------------------
    # Kalman filter — predict where the target will be after servo lag.
    # -----------------------------------------------------------------------
    now = time.monotonic()
    dt = now - _last_aim_time if _last_aim_time > 0.0 else 0.033  # assume ~30 fps on first call
    _last_aim_time = now

    lookahead_s = servo_kalman_lookahead_ms.value / 1000.0
    predicted_pan, predicted_tilt = _kalman.update(
        pan_angle, tilt_angle, dt,
        process_noise=servo_kalman_process_noise.value,
        measurement_noise=servo_kalman_meas_noise.value,
        lookahead_s=lookahead_s,
    )
    logger.debug(
        "Kalman: raw=(%.1f°, %.1f°) predicted=(%.1f°, %.1f°) lookahead=%.0fms",
        pan_angle, tilt_angle, predicted_pan, predicted_tilt, lookahead_s * 1000,
    )

    # Reset Kalman on large jumps (new target or tracking lost), same threshold as PID.
    if abs(pan_angle - servo_pan.value) > _RESET_THRESHOLD or abs(tilt_angle - servo_tilt.value) > _RESET_THRESHOLD:
        _kalman.reset()
        predicted_pan, predicted_tilt = pan_angle, tilt_angle

    # Publish predicted position so the stream renderer can draw a crosshair.
    servo_kalman_pan.value = predicted_pan
    servo_kalman_tilt.value = predicted_tilt

    # -----------------------------------------------------------------------
    # Move servos via hardware PWM (Hobot.GPIO).
    # -----------------------------------------------------------------------
    pan_clamped = max(-90.0, min(90.0, predicted_pan))
    tilt_clamped = max(-90.0, min(90.0, predicted_tilt))

    # Dead zone: skip updates where both axes haven't moved enough to matter.
    # This prevents micro-jitter caused by detection noise around a stable target.
    dead_zone = servo_dead_zone.value
    if abs(pan_clamped - servo_pan.value) < dead_zone and abs(tilt_clamped - servo_tilt.value) < dead_zone:
        logger.debug(
            "Servo update suppressed by dead zone (Δpan=%.2f° Δtilt=%.2f° < %.2f°)",
            abs(pan_clamped - servo_pan.value),
            abs(tilt_clamped - servo_tilt.value),
            dead_zone,
        )
        return pan_angle, tilt_angle

    # PID controller: compute the next commanded angle for each axis.
    # Kp=1, Ki=0, Kd=0 reproduces the previous instant-snap behaviour.
    # Raise Kd (e.g. 0.1–0.2) to dampen detection-noise jitter.
    pan_new, tilt_new = _pid_step(
        pan_clamped, tilt_clamped,
        servo_pan.value, servo_tilt.value,
        servo_pid_kp.value, servo_pid_ki.value, servo_pid_kd.value,
    )

    # Read invert flags and vertical offset from the shared aim settings.
    pan_inv = app_settings.aim_settings.pan_invert
    tilt_inv = app_settings.aim_settings.tilt_invert

    from ..utils.shared import vertical_angle_offset  # noqa: PLC0415
    tilt_offset = vertical_angle_offset.value

    pan_pwm = _get_pan_pwm()
    if pan_pwm is not None:
        try:
            # By default pan is negated (physical mounting).  If pan_invert is
            # set, flip the sign so the servo moves the other way.
            pan_dc = -pan_new if not pan_inv else pan_new
            pan_pwm.ChangeDutyCycle(_angle_to_dc(pan_dc))
            logger.info("Pan servo → %.1f° (target %.1f°)", pan_new, pan_clamped)
        except Exception:
            logger.exception("Failed to move pan servo")

    tilt_pwm = _get_tilt_pwm()
    if tilt_pwm is not None:
        try:
            # Apply vertical angle offset at the output so the crosshair shows
            # the raw target position but the servo compensates for mounting height.
            tilt_out = (tilt_new + tilt_offset) if not tilt_inv else -(tilt_new + tilt_offset)
            tilt_pwm.ChangeDutyCycle(_angle_to_dc(tilt_out))
            logger.info("Tilt servo → %.1f° (target %.1f° offset=%.1f°)", tilt_out, tilt_clamped, tilt_offset)
        except Exception:
            logger.exception("Failed to move tilt servo")

    servo_pan.value = pan_new
    servo_tilt.value = tilt_new

    return pan_angle, tilt_angle


def move_to(pan_angle: float, tilt_angle: float) -> tuple[float, float]:
    """Directly command both servos to the given angles (manual / debug mode).

    Args:
        pan_angle:  Desired pan angle in degrees, clamped to ``[-90, 90]``.
                    Positive = right of centre.
        tilt_angle: Desired tilt angle in degrees, clamped to ``[-90, 90]``.
                    Positive = down from centre.

    Returns:
        ``(pan_clamped, tilt_clamped)`` — the angles actually sent to the servos.
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
    """Activate the splash relay for *duration_s* seconds, non-blocking.

    The relay is deactivated by a daemon thread after the duration elapses.
    Safe to call from any thread.
    """
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
