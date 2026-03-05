import logging
import os
import traceback
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera geometry — override via environment variables for your hardware.
# ---------------------------------------------------------------------------
# Horizontal / vertical field of view of the camera lens (degrees).
CAMERA_HFOV: float = float(os.environ.get("CAMERA_HFOV", "62.0"))
CAMERA_VFOV: float = float(os.environ.get("CAMERA_VFOV", "48.0"))

# Rectified camera intrinsics published by hobot_stereonet for 640×352 images.
# These can be overridden if the calibration changes.
CAM_FX: float = float(os.environ.get("CAMERA_FX", "257.854584"))
CAM_FY: float = float(os.environ.get("CAMERA_FY", "257.854584"))
CAM_CX: float = float(os.environ.get("CAMERA_CX", "314.431396"))
CAM_CY: float = float(os.environ.get("CAMERA_CY", "159.979248"))

# GPIO pin connected to the pan servo.
SERVO_PAN_PIN: int = int(os.environ.get("SERVO_PAN_PIN", "12"))

# ---------------------------------------------------------------------------
# Servo — initialised lazily so that import failures don't crash the app.
# ---------------------------------------------------------------------------
_servo = None


def _get_servo():
    global _servo  # noqa: PLW0603
    if _servo is not None:
        return _servo
    try:
        from gpiozero import AngularServo  # noqa: PLC0415
        from gpiozero.pins.pigpio import PiGPIOFactory  # noqa: PLC0415

        factory = PiGPIOFactory()
        _servo = AngularServo(
            SERVO_PAN_PIN,
            pin_factory=factory,
            min_angle=-90,
            max_angle=90,
        )
        logger.info("Pan servo initialised on GPIO pin %d", SERVO_PAN_PIN)
    except Exception:
        logger.warning("Servo unavailable — aim_at() will log only", exc_info=True)
        _servo = None
    return _servo


def aim_at(
    bbox_normalized: list[float],
    depth_map: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Map a detected object's bounding box to a physical pan angle and move the servo.

    Args:
        bbox_normalized: Detection bounding box as ``[ymin, xmin, ymax, xmax]``
                         with all values normalised to ``[0, 1]`` relative to the
                         640×640 model input (which may be zero-padded at the bottom).
        depth_map:       Optional mono16 depth image (H×W, values in mm) from
                         ``/StereoNetNode/stereonet_depth``.  When provided the cat's
                         3-D position (X, Y, Z in metres) is computed and logged.

    Returns:
        ``(pan_angle, tilt_angle)`` in degrees (positive = right / down).
    """
    ymin, xmin, ymax, xmax = bbox_normalized

    # Normalised centre of the detection box.
    cx_n = (xmin + xmax) / 2.0   # 0 = left, 1 = right
    cy_n = (ymin + ymax) / 2.0   # 0 = top,  1 = bottom

    # Angle offset from the optical centre (degrees).
    pan_angle: float = (cx_n - 0.5) * CAMERA_HFOV
    tilt_angle: float = (cy_n - 0.5) * CAMERA_VFOV

    # -----------------------------------------------------------------------
    # 3-D localisation via stereo depth map.
    # bbox_normalized is relative to the 640×640 padded model input; the
    # actual camera image is 640×352 so cy_n may address padding rows.
    # -----------------------------------------------------------------------
    if depth_map is not None:
        dh, dw = depth_map.shape
        # Map normalised coordinates back to depth-map pixel space.
        px = int(np.clip(cx_n * dw, 0, dw - 1))
        py = int(np.clip(cy_n * dh, 0, dh - 1))          # clamps to valid rows

        z_mm = float(depth_map[py, px])
        if z_mm > 0:
            Z = z_mm / 1000.0                             # metres
            X = (px - CAM_CX) * Z / CAM_FX
            Y = (py - CAM_CY) * Z / CAM_FY
            logger.info(
                "Cat 3-D position: X=%.2fm Y=%.2fm Z=%.2fm  (pan=%.1f° tilt=%.1f°)",
                X, Y, Z, pan_angle, tilt_angle,
            )
        else:
            logger.info(
                "Cat at pan=%.1f° tilt=%.1f° (depth unavailable at pixel %d,%d)",
                pan_angle, tilt_angle, px, py,
            )
    else:
        logger.info(
            "Cat at pan=%.1f° tilt=%.1f° (no depth map)",
            pan_angle, tilt_angle,
        )

    servo = _get_servo()
    if servo is not None:
        try:
            clamped = max(-90.0, min(90.0, pan_angle))
            servo.angle = clamped
            logger.info("Servo moved to %.1f°", clamped)
        except Exception:
            logger.exception("Failed to move servo")

    return pan_angle, tilt_angle
