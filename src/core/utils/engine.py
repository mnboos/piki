import atexit
import logging
import math
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera geometry — override via environment variables for your hardware.
# ---------------------------------------------------------------------------
# Rectified camera intrinsics published by hobot_stereonet for 640×352 images.
# These are used with the atan2 pinhole formula for accurate angle mapping.
# The stereonet ROS node already removes fisheye distortion; we work with the
# rectified output only.
CAM_FX: float = float(os.environ.get("CAMERA_FX", "257.854584"))
CAM_FY: float = float(os.environ.get("CAMERA_FY", "257.854584"))
CAM_CX: float = float(os.environ.get("CAMERA_CX", "314.431396"))
CAM_CY: float = float(os.environ.get("CAMERA_CY", "159.979248"))

# Actual output frame dimensions from the stereonet node (pixels).
# With the intrinsics above the effective HFOV ≈ 102° and VFOV ≈ 68°.
FRAME_W: int = int(os.environ.get("CAMERA_FRAME_W", "640"))
FRAME_H: int = int(os.environ.get("CAMERA_FRAME_H", "352"))

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

    Uses the pinhole camera model with calibrated intrinsics for accurate
    angle mapping of the rectified stereonet output image.

    Args:
        bbox_normalized: ``[ymin, xmin, ymax, xmax]`` normalised to ``[0, 1]``
                         relative to the actual frame (FRAME_W × FRAME_H).

    Returns:
        ``(pan_angle, tilt_angle)`` in degrees.  Positive pan = right of centre,
        positive tilt = below centre.
    """
    ymin, xmin, ymax, xmax = bbox_normalized

    # Normalised centre of the detection box → pixel coordinates.
    cx_n = (xmin + xmax) / 2.0
    cy_n = (ymin + ymax) / 2.0
    px = cx_n * FRAME_W
    py = cy_n * FRAME_H

    # Pinhole atan2 formula — more accurate than a linear FOV mapping because
    # the angle is non-linear even for a perfect pinhole lens.
    pan_angle = math.degrees(math.atan2(px - CAM_CX, CAM_FX))
    tilt_angle = math.degrees(math.atan2(py - CAM_CY, CAM_FY))

    return pan_angle, tilt_angle


def aim_at(
    bbox_normalized: list[float],
    depth_map: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Aim both pan and tilt servos at a detected object.

    Args:
        bbox_normalized: Detection bounding box as ``[ymin, xmin, ymax, xmax]``
                         normalised to ``[0, 1]`` relative to the actual frame.
        depth_map:       Optional mono16 depth image (H×W, values in mm) from
                         ``/StereoNetNode/stereonet_depth``.  When provided the
                         object's 3-D position (X, Y, Z in metres) is logged.

    Returns:
        ``(pan_angle, tilt_angle)`` in degrees (positive = right / down).
    """
    pan_angle, tilt_angle = bbox_to_angles(bbox_normalized)

    # -----------------------------------------------------------------------
    # Optional 3-D localisation via stereo depth map.
    # -----------------------------------------------------------------------
    if depth_map is not None:
        dh, dw = depth_map.shape
        cx_n = (bbox_normalized[1] + bbox_normalized[3]) / 2.0
        cy_n = (bbox_normalized[0] + bbox_normalized[2]) / 2.0
        px = int(np.clip(cx_n * dw, 0, dw - 1))
        py = int(np.clip(cy_n * dh, 0, dh - 1))

        z_mm = float(depth_map[py, px])
        if z_mm > 0:
            Z = z_mm / 1000.0
            X = (px - CAM_CX) * Z / CAM_FX
            Y = (py - CAM_CY) * Z / CAM_FY
            logger.info(
                "Target 3-D position: X=%.2fm Y=%.2fm Z=%.2fm  (pan=%.1f° tilt=%.1f°)",
                X, Y, Z, pan_angle, tilt_angle,
            )
        else:
            logger.info(
                "Target at pan=%.1f° tilt=%.1f° (depth unavailable at pixel %d,%d)",
                pan_angle, tilt_angle, px, py,
            )
    else:
        logger.info("Target at pan=%.1f° tilt=%.1f° (no depth map)", pan_angle, tilt_angle)

    # -----------------------------------------------------------------------
    # Move servos via hardware PWM (Hobot.GPIO).
    # -----------------------------------------------------------------------
    pan_clamped = max(-90.0, min(90.0, pan_angle))
    tilt_clamped = max(-90.0, min(90.0, tilt_angle))

    pan_pwm = _get_pan_pwm()
    if pan_pwm is not None:
        try:
            pan_pwm.ChangeDutyCycle(_angle_to_dc(-pan_clamped))
            logger.info("Pan servo → %.1f°", pan_clamped)
        except Exception:
            logger.exception("Failed to move pan servo")

    tilt_pwm = _get_tilt_pwm()
    if tilt_pwm is not None:
        try:
            tilt_pwm.ChangeDutyCycle(_angle_to_dc(tilt_clamped))
            logger.info("Tilt servo → %.1f°", tilt_clamped)
        except Exception:
            logger.exception("Failed to move tilt servo")

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

    return pan_clamped, tilt_clamped
