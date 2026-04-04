import atexit
import logging
import os
from typing import Optional

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

# Standard 50 Hz servo PWM: 1.5 ms centre pulse → 7.5% duty cycle.
# Mapping: angle [-90°, +90°] → duty cycle [2.5%, 12.5%]
_SERVO_FREQ_HZ = 50
_DC_CENTER = 7.5
_DC_RANGE = 5.0  # ±5% spans ±90°


def _angle_to_dc(angle: float) -> float:
    """Convert servo angle (−90…+90°) to PWM duty cycle (2.5…12.5%)."""
    return _DC_CENTER + (angle / 90.0) * _DC_RANGE


# ---------------------------------------------------------------------------
# Servos — initialised lazily so that import failures don't crash the app.
# Hobot.GPIO is the correct GPIO library for the RDK X5 (RPi.GPIO-compatible).
# ---------------------------------------------------------------------------
_pan_pwm = None
_tilt_pwm = None
_gpio_initialised = False


@atexit.register
def _cleanup_gpio() -> None:
    global _pan_pwm, _tilt_pwm  # noqa: PLW0603
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
            GPIO.cleanup([SERVO_PAN_PIN, SERVO_TILT_PIN])
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
    depth_map: "Optional[object]" = None,
) -> tuple[float, float]:
    """Aim both pan and tilt servos at a detected object.

    Args:
        bbox_normalized: Detection bounding box as ``[ymin, xmin, ymax, xmax]``
                         normalised to ``[0, 1]`` relative to the actual frame.
        depth_map:       Ignored — kept for call-site compatibility.

    Returns:
        ``(pan_angle, tilt_angle)`` in degrees (positive = right / down).
    """
    pan_angle, tilt_angle = bbox_to_angles(bbox_normalized)
    logger.debug("Target at pan=%.1f° tilt=%.1f°", pan_angle, tilt_angle)

    # -----------------------------------------------------------------------
    # Move servos via hardware PWM (Hobot.GPIO).
    # -----------------------------------------------------------------------
    pan_clamped = max(-90.0, min(90.0, pan_angle))
    tilt_clamped = max(-90.0, min(90.0, tilt_angle))

    from ..utils.shared import servo_dead_zone, servo_pan, servo_smooth_factor, servo_tilt  # noqa: PLC0415

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

    # EMA smoothing: blend the new target with the current position so the servo
    # glides toward the target rather than snapping instantly.
    # alpha=1.0 → instant (original behaviour); alpha≈0.3 → heavy smoothing.
    alpha = servo_smooth_factor.value
    pan_smoothed = alpha * pan_clamped + (1.0 - alpha) * servo_pan.value
    tilt_smoothed = alpha * tilt_clamped + (1.0 - alpha) * servo_tilt.value

    pan_pwm = _get_pan_pwm()
    if pan_pwm is not None:
        try:
            pan_pwm.ChangeDutyCycle(_angle_to_dc(-pan_smoothed))
            logger.info("Pan servo → %.1f° (target %.1f°)", pan_smoothed, pan_clamped)
        except Exception:
            logger.exception("Failed to move pan servo")

    tilt_pwm = _get_tilt_pwm()
    if tilt_pwm is not None:
        try:
            tilt_pwm.ChangeDutyCycle(_angle_to_dc(tilt_smoothed))
            logger.info("Tilt servo → %.1f° (target %.1f°)", tilt_smoothed, tilt_clamped)
        except Exception:
            logger.exception("Failed to move tilt servo")

    servo_pan.value = pan_smoothed
    servo_tilt.value = tilt_smoothed

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

    pan_pwm = _get_pan_pwm()
    if pan_pwm is not None:
        try:
            pan_pwm.ChangeDutyCycle(_angle_to_dc(-pan_clamped))
        except Exception:
            logger.exception("Failed to move pan servo")

    tilt_pwm = _get_tilt_pwm()
    if tilt_pwm is not None:
        try:
            tilt_pwm.ChangeDutyCycle(_angle_to_dc(tilt_clamped))
        except Exception:
            logger.exception("Failed to move tilt servo")

    from ..utils.shared import servo_pan, servo_tilt  # noqa: PLC0415
    servo_pan.value = pan_clamped
    servo_tilt.value = tilt_clamped

    return pan_clamped, tilt_clamped
