import asyncio
import math
from typing import Optional

import numpy as np
from django.http import HttpRequest
from django.http.response import StreamingHttpResponse
from ninja import NinjaAPI, PatchDict, Schema

from .utils.shared import (
    app_settings,
    cv2,
    is_object_detection_disabled,
    latest_frame,
    mask_transparency,
    motion_detector,
    prob_threshold,
    servo_pan,
    servo_tilt,
    settings,
    streaming_active,
)

# api = NinjaAPI(csrf=True, auth=django_auth)
api = NinjaAPI()

# All 80 trimmed COCO class names (same order as ai.py CLASSES tuple).
_YOLO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


async def stream_camera():
    """Video streaming generator function with corrected drawing logic."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    box_color = (0, 255, 128)  # A nice green for the boxes
    thickness = 2

    is_object_detection_disabled.clear()
    streaming_active.set()
    try:
        last_ts = 0
        while True:
            # Wait for a new frame from the producer thread
            result = await asyncio.to_thread(latest_frame.wait_for_frame, last_ts)
            if result is None:
                await asyncio.sleep(0.01)
                continue

            frame, detections, last_ts = result

            if frame is None or (hasattr(frame, "size") and frame.size == 0):
                continue

            # Convert grayscale Y-plane (2D) to writable BGR for drawing.
            # frame_lores is the decimated NV12 Y-plane — single-channel uint8.
            if frame.ndim == 2:
                draw_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                draw_frame = np.array(frame)  # writable copy if already BGR

            # Draw detections on the frame
            for detection in detections:
                left, top, w, h = detection.bbox
                left, top, w, h = int(left), int(top), int(w), int(h)
                right = left + w
                bottom = top + h

                cv2.rectangle(draw_frame, (left, top), (right, bottom), box_color, thickness)

                text_to_draw = f"{detection.label} ({detection.confidence:.1%})"
                (text_w, text_h), _ = cv2.getTextSize(text_to_draw, font, font_scale, thickness)
                text_bg_rect_start = (left, top - text_h - 7)
                text_bg_rect_end = (left + text_w, top)
                cv2.rectangle(draw_frame, text_bg_rect_start, text_bg_rect_end, box_color, -1)

                cv2.putText(
                    draw_frame,
                    text_to_draw,
                    (left, top - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Draw servo crosshair using camera intrinsics from engine.py.
            # angle→pixel: px = cx + fx*tan(pan), py = cy + fy*tan(tilt)
            from .utils.engine import CAM_CX, CAM_CY, CAM_FX, CAM_FY, FRAME_H, FRAME_W  # noqa: PLC0415
            fh, fw = draw_frame.shape[:2]
            scale_x = fw / FRAME_W
            scale_y = fh / FRAME_H
            ch_x = int((CAM_CX + CAM_FX * math.tan(math.radians(servo_pan.value))) * scale_x)
            ch_y = int((CAM_CY + CAM_FY * math.tan(math.radians(servo_tilt.value))) * scale_y)
            ch_x = max(0, min(fw - 1, ch_x))
            ch_y = max(0, min(fh - 1, ch_y))
            _CROSSHAIR_COLOR = (0, 200, 255)  # orange
            _CROSSHAIR_RADIUS = 14
            _CROSSHAIR_GAP = 4
            _CROSSHAIR_THICKNESS = 2
            # Four line segments around the centre with a gap
            cv2.line(draw_frame, (ch_x, ch_y - _CROSSHAIR_GAP), (ch_x, ch_y - _CROSSHAIR_RADIUS), _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS)
            cv2.line(draw_frame, (ch_x, ch_y + _CROSSHAIR_GAP), (ch_x, ch_y + _CROSSHAIR_RADIUS), _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS)
            cv2.line(draw_frame, (ch_x - _CROSSHAIR_GAP, ch_y), (ch_x - _CROSSHAIR_RADIUS, ch_y), _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS)
            cv2.line(draw_frame, (ch_x + _CROSSHAIR_GAP, ch_y), (ch_x + _CROSSHAIR_RADIUS, ch_y), _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS)
            cv2.circle(draw_frame, (ch_x, ch_y), _CROSSHAIR_GAP, _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            success, buffer = cv2.imencode(".jpeg", draw_frame, encode_param)
            if success:
                frame_bytes = buffer.tobytes()
                yield b"--frame\nContent-Type: image/jpeg\n\n" + frame_bytes + b"\n"
    finally:
        streaming_active.clear()


BIN_RESPONSE = {
    "responses": {
        200: {
            "description": "OK",
            "content": {
                "multipart/x-mixed-replace; boundary=frame": {"schema": {"type": "string", "format": "binary"}},
            },
        },
    },
}


@api.get("/video_feed", openapi_extra=BIN_RESPONSE)
async def video_feed(request: HttpRequest):
    """Video streaming route."""
    return StreamingHttpResponse(stream_camera(), content_type="multipart/x-mixed-replace; boundary=frame")


class PikiOptions(Schema):
    mode: str
    conf_threshold: Optional[float] = None
    pixelcount_threshold: Optional[int] = None
    min_area: Optional[int] = None
    mog2_history: Optional[int] = None
    mog2_var_threshold: Optional[int] = None
    denoise_kernelsize: Optional[int] = None
    mask_transparency: Optional[float] = None


@api.patch("/update_options", response=PikiOptions)
def update_options(request: HttpRequest, options: PatchDict[PikiOptions]):
    from .models import DetectionConfig  # noqa: PLC0415

    mode = options.get("mode", app_settings.debug_settings.mode)
    app_settings.debug_settings.mode = mode
    # debug_enabled gates the mask/rois branch in process_frame
    app_settings.debug_settings.debug_enabled = mode in ("mask", "rois")

    if (v := options.get("conf_threshold")) is not None:
        prob_threshold.value = float(v)

    if (v := options.get("pixelcount_threshold")) is not None:
        settings.foreground_mask_options.pixelcount_threshold.value = int(v)

    if (v := options.get("min_area")) is not None:
        settings.foreground_mask_options.min_area.value = int(v)

    if (v := options.get("mog2_history")) is not None:
        settings.foreground_mask_options.mog2_history.value = int(v)

    if (v := options.get("mog2_var_threshold")) is not None:
        settings.foreground_mask_options.mog2_var_threshold.value = int(v)

    if (v := options.get("denoise_kernelsize")) is not None:
        settings.foreground_mask_options.denoise_kernelsize.value = int(v)

    if (v := options.get("mask_transparency")) is not None:
        mask_transparency.value = float(v)

    # Persist all current values to DB so they survive restarts.
    config = DetectionConfig.load()
    config.mode = mode
    config.conf_threshold = prob_threshold.value
    config.pixelcount_threshold = settings.foreground_mask_options.pixelcount_threshold.value
    config.min_area = settings.foreground_mask_options.min_area.value
    config.mog2_history = settings.foreground_mask_options.mog2_history.value
    config.mog2_var_threshold = settings.foreground_mask_options.mog2_var_threshold.value
    config.denoise_kernelsize = settings.foreground_mask_options.denoise_kernelsize.value
    config.mask_transparency = mask_transparency.value
    config.save()

    return PikiOptions(
        mode=mode,
        conf_threshold=prob_threshold.value,
        pixelcount_threshold=settings.foreground_mask_options.pixelcount_threshold.value,
        min_area=settings.foreground_mask_options.min_area.value,
        mog2_history=settings.foreground_mask_options.mog2_history.value,
        mog2_var_threshold=settings.foreground_mask_options.mog2_var_threshold.value,
        denoise_kernelsize=settings.foreground_mask_options.denoise_kernelsize.value,
        mask_transparency=mask_transparency.value,
    )


@api.post("/reset_background")
def reset_background(request: HttpRequest):
    """Discard the MOG2 background model so it relearns the current scene."""
    motion_detector.reset()
    return {"status": "ok"}


@api.get("/options", response=PikiOptions)
def get_options(request: HttpRequest):
    """Return current tuning values so the frontend can initialise its controls."""
    return PikiOptions(
        mode=app_settings.debug_settings.mode or "boxes",
        conf_threshold=prob_threshold.value,
        pixelcount_threshold=settings.foreground_mask_options.pixelcount_threshold.value,
        min_area=settings.foreground_mask_options.min_area.value,
        mog2_history=settings.foreground_mask_options.mog2_history.value,
        mog2_var_threshold=settings.foreground_mask_options.mog2_var_threshold.value,
        denoise_kernelsize=settings.foreground_mask_options.denoise_kernelsize.value,
        mask_transparency=mask_transparency.value,
    )


class AimConfigSchema(Schema):
    target_classes: list[str]
    servo_enabled: bool


@api.get("/aim_config", response=AimConfigSchema)
def get_aim_config(request: HttpRequest):
    """Return current servo aim configuration."""
    return AimConfigSchema(
        target_classes=list(app_settings.aim_settings.target_classes or []),
        servo_enabled=bool(app_settings.aim_settings.servo_enabled),
    )


@api.patch("/aim_config", response=AimConfigSchema)
def update_aim_config(request: HttpRequest, payload: PatchDict[AimConfigSchema]):
    """Update servo aim configuration and persist to database."""
    from .models import AimConfig  # noqa: PLC0415

    config = AimConfig.load()

    if (classes := payload.get("target_classes")) is not None:
        validated = [str(c).strip() for c in classes if str(c).strip() in _YOLO_CLASSES]
        app_settings.aim_settings.target_classes = validated
        config.target_classes = validated

    if (enabled := payload.get("servo_enabled")) is not None:
        app_settings.aim_settings.servo_enabled = bool(enabled)
        config.servo_enabled = bool(enabled)

    config.save()

    return AimConfigSchema(
        target_classes=list(app_settings.aim_settings.target_classes or []),
        servo_enabled=bool(app_settings.aim_settings.servo_enabled),
    )


@api.get("/yolo_classes", response=list[str])
def get_yolo_classes(request: HttpRequest):
    """Return the list of all detectable YOLO class names."""
    return _YOLO_CLASSES


class ServoMoveSchema(Schema):
    pan_angle: float   # degrees, -90 (left) … +90 (right)
    tilt_angle: float  # degrees, -90 (up)   … +90 (down)


class ServoPositionSchema(Schema):
    pan_angle: float
    tilt_angle: float


@api.post("/servo/move", response=ServoPositionSchema)
def servo_move(request: HttpRequest, payload: ServoMoveSchema):
    """Manually command both servos to explicit angles (debug / calibration mode)."""
    from .utils.engine import move_to  # noqa: PLC0415

    pan, tilt = move_to(payload.pan_angle, payload.tilt_angle)
    return ServoPositionSchema(pan_angle=pan, tilt_angle=tilt)

