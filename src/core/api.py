import asyncio
from typing import Optional

import numpy as np
from django.http import HttpRequest
from django.http.response import StreamingHttpResponse
from ninja import NinjaAPI, PatchDict, Schema

from .utils.shared import (
    app_settings,
    bbox_ema_alpha,
    cv2,
    event_clip_queue,
    event_clip_queue_lock,
    event_cooldown_seconds,
    event_post_trigger_seconds,
    event_pre_buffer_seconds,
    event_recording_active,
    event_recording_cooldown_until,
    event_recording_enabled,
    event_trigger_classes,
    event_trigger_classes_lock,
    fps_counter,
    ghost_frames_ms,
    is_object_detection_disabled,
    latest_debug_frame,
    latest_frame,
    mask_transparency,
    min_consecutive_frames,
    motion_detector,
    prob_threshold,
    prob_threshold_keep,
    recording_active,
    replaying_active,
    servo_dead_zone,
    servo_kalman_meas_noise,
    servo_kalman_process_noise,
    servo_pan,
    servo_pid_kd,
    servo_pid_ki,
    servo_pid_kp,
    servo_tilt,
    settings,
    pump_duty,
    splash_cooldown,
    splash_delay,
    splash_duration,
    splash_enabled,
    streaming_active,
    tracker_confirm_hits,
    tracker_enabled,
    tracker_iou_threshold,
    tracker_max_misses,
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
            if app_settings.debug_settings.show_boxes:
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

            # Draw servo crosshair using the same linear FOV model as bbox_to_angles.
            # Inverse: cx_n = pan / HFOV + 0.5  →  px = cx_n * frame_width
            from .utils.engine import SERVO_HFOV, SERVO_VFOV  # noqa: PLC0415
            from .utils.shared import servo_kalman_pan, servo_kalman_tilt  # noqa: PLC0415
            fh, fw = draw_frame.shape[:2]

            def _angle_to_px(pan: float, tilt: float) -> tuple[int, int]:
                x = max(0, min(fw - 1, int((pan / SERVO_HFOV + 0.5) * fw)))
                y = max(0, min(fh - 1, int((tilt / SERVO_VFOV + 0.5) * fh)))
                return x, y

            _CROSSHAIR_RADIUS = 14
            _CROSSHAIR_GAP = 4
            _CROSSHAIR_THICKNESS = 2

            # Kalman prediction crosshair (cyan, smaller, no gap circle — just tick marks)
            kp_x, kp_y = _angle_to_px(servo_kalman_pan.value, servo_kalman_tilt.value)
            _KP_COLOR = (255, 220, 0)  # cyan
            _KP_RADIUS = 10
            cv2.line(draw_frame, (kp_x, kp_y - _CROSSHAIR_GAP), (kp_x, kp_y - _KP_RADIUS), _KP_COLOR, _CROSSHAIR_THICKNESS)
            cv2.line(draw_frame, (kp_x, kp_y + _CROSSHAIR_GAP), (kp_x, kp_y + _KP_RADIUS), _KP_COLOR, _CROSSHAIR_THICKNESS)
            cv2.line(draw_frame, (kp_x - _CROSSHAIR_GAP, kp_y), (kp_x - _KP_RADIUS, kp_y), _KP_COLOR, _CROSSHAIR_THICKNESS)
            cv2.line(draw_frame, (kp_x + _CROSSHAIR_GAP, kp_y), (kp_x + _KP_RADIUS, kp_y), _KP_COLOR, _CROSSHAIR_THICKNESS)

            # Current servo position crosshair (orange, larger, with centre dot)
            ch_x, ch_y = _angle_to_px(servo_pan.value, servo_tilt.value)
            _CROSSHAIR_COLOR = (0, 200, 255)  # orange
            cv2.line(draw_frame, (ch_x, ch_y - _CROSSHAIR_GAP), (ch_x, ch_y - _CROSSHAIR_RADIUS), _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS)
            cv2.line(draw_frame, (ch_x, ch_y + _CROSSHAIR_GAP), (ch_x, ch_y + _CROSSHAIR_RADIUS), _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS)
            cv2.line(draw_frame, (ch_x - _CROSSHAIR_GAP, ch_y), (ch_x - _CROSSHAIR_RADIUS, ch_y), _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS)
            cv2.line(draw_frame, (ch_x + _CROSSHAIR_GAP, ch_y), (ch_x + _CROSSHAIR_RADIUS, ch_y), _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS)
            cv2.circle(draw_frame, (ch_x, ch_y), _CROSSHAIR_GAP, _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS)

            # Line connecting servo position to Kalman prediction (shows lead distance)
            if abs(kp_x - ch_x) > 3 or abs(kp_y - ch_y) > 3:
                cv2.line(draw_frame, (ch_x, ch_y), (kp_x, kp_y), (180, 180, 180), 1, cv2.LINE_AA)

            # Exclusion zones — always visible (safety-critical, not optional).
            # Drawn last so they sit on top of detection boxes and the crosshair.
            from .utils import exclusion as _exclusion  # noqa: PLC0415

            zone_polys = _exclusion.polygons_norm()
            if zone_polys:
                _zone_color = (60, 60, 220)  # dark red (BGR)
                overlay = draw_frame.copy()
                pts_int = [
                    np.round(p * np.array([fw, fh], dtype=np.float32)).astype(np.int32)
                    for p in zone_polys
                ]
                cv2.fillPoly(overlay, pts_int, color=_zone_color)
                cv2.addWeighted(overlay, 0.30, draw_frame, 0.70, 0, draw_frame)
                cv2.polylines(draw_frame, pts_int, isClosed=True, color=_zone_color, thickness=2)

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


async def stream_debug_camera():
    """MJPEG stream of the raw/debug topic — no overlays, no detection boxes."""
    last_ts = 0
    try:
        while True:
            result = await asyncio.to_thread(latest_debug_frame.wait_for_frame, last_ts)
            if result is None:
                await asyncio.sleep(0.01)
                continue
            frame, _, last_ts = result
            if frame is None or (hasattr(frame, "size") and frame.size == 0):
                continue
            draw_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else np.array(frame)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            success, buffer = cv2.imencode(".jpeg", draw_frame, encode_param)
            if success:
                yield b"--frame\nContent-Type: image/jpeg\n\n" + buffer.tobytes() + b"\n"
    finally:
        pass


@api.get("/video_feed_raw", openapi_extra=BIN_RESPONSE, operation_id="video_feed_debug")
async def video_feed_raw(request: HttpRequest):
    """Raw/debug video feed — streams the ROS_DEBUG_TOPIC without any overlays."""
    return StreamingHttpResponse(stream_debug_camera(), content_type="multipart/x-mixed-replace; boundary=frame")


class PikiOptions(Schema):
    show_boxes: bool = True
    show_mask: bool = False
    show_rois: bool = False
    conf_threshold: Optional[float] = None
    conf_threshold_keep: Optional[float] = None
    min_consecutive_frames: Optional[int] = None
    bbox_ema_alpha: Optional[float] = None
    ghost_frames_ms: Optional[int] = None
    tracker_enabled: Optional[bool] = None
    tracker_iou_threshold: Optional[float] = None
    tracker_max_misses: Optional[int] = None
    tracker_confirm_hits: Optional[int] = None
    pixelcount_threshold: Optional[int] = None
    min_area: Optional[int] = None
    mog2_history: Optional[int] = None
    mog2_var_threshold: Optional[int] = None
    denoise_kernelsize: Optional[int] = None
    mask_transparency: Optional[float] = None
    servo_pid_kp: Optional[float] = None
    servo_pid_ki: Optional[float] = None
    servo_pid_kd: Optional[float] = None
    servo_dead_zone: Optional[float] = None
    servo_kalman_process_noise: Optional[float] = None
    servo_kalman_meas_noise: Optional[float] = None


@api.patch("/update_options", response=PikiOptions)
def update_options(request: HttpRequest, options: PatchDict[PikiOptions]):
    from .models import DetectionConfig  # noqa: PLC0415

    if (v := options.get("show_boxes")) is not None:
        app_settings.debug_settings.show_boxes = v
    if (v := options.get("show_mask")) is not None:
        app_settings.debug_settings.show_mask = v
    if (v := options.get("show_rois")) is not None:
        app_settings.debug_settings.show_rois = v

    if (v := options.get("conf_threshold")) is not None:
        prob_threshold.value = float(v)
        # Keep the "keep" floor strictly ≤ the "enter" threshold so hysteresis
        # remains well-formed when only the enter value is adjusted.
        if prob_threshold_keep.value > prob_threshold.value:
            prob_threshold_keep.value = prob_threshold.value

    if (v := options.get("conf_threshold_keep")) is not None:
        prob_threshold_keep.value = min(float(v), float(prob_threshold.value))

    if (v := options.get("min_consecutive_frames")) is not None:
        min_consecutive_frames.value = max(1, int(v))

    if (v := options.get("bbox_ema_alpha")) is not None:
        bbox_ema_alpha.value = max(0.0, min(1.0, float(v)))

    if (v := options.get("ghost_frames_ms")) is not None:
        ghost_frames_ms.value = max(0, int(v))

    if (v := options.get("tracker_enabled")) is not None:
        tracker_enabled.value = 1 if v else 0

    if (v := options.get("tracker_iou_threshold")) is not None:
        tracker_iou_threshold.value = max(0.0, min(1.0, float(v)))

    if (v := options.get("tracker_max_misses")) is not None:
        tracker_max_misses.value = max(0, int(v))

    if (v := options.get("tracker_confirm_hits")) is not None:
        tracker_confirm_hits.value = max(1, int(v))

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

    if (v := options.get("servo_pid_kp")) is not None:
        servo_pid_kp.value = max(0.0, float(v))

    if (v := options.get("servo_pid_ki")) is not None:
        servo_pid_ki.value = max(0.0, float(v))

    if (v := options.get("servo_pid_kd")) is not None:
        servo_pid_kd.value = max(0.0, float(v))

    if (v := options.get("servo_dead_zone")) is not None:
        servo_dead_zone.value = max(0.0, float(v))

    if (v := options.get("servo_kalman_process_noise")) is not None:
        servo_kalman_process_noise.value = max(0.01, float(v))

    if (v := options.get("servo_kalman_meas_noise")) is not None:
        servo_kalman_meas_noise.value = max(0.01, float(v))

    # Persist all current values to DB so they survive restarts.
    config = DetectionConfig.load()
    config.show_boxes = app_settings.debug_settings.show_boxes
    config.show_mask = app_settings.debug_settings.show_mask
    config.show_rois = app_settings.debug_settings.show_rois
    config.conf_threshold = prob_threshold.value
    config.conf_threshold_keep = prob_threshold_keep.value
    config.min_consecutive_frames = min_consecutive_frames.value
    config.bbox_ema_alpha = bbox_ema_alpha.value
    config.ghost_frames_ms = ghost_frames_ms.value
    config.tracker_enabled = bool(tracker_enabled.value)
    config.tracker_iou_threshold = tracker_iou_threshold.value
    config.tracker_max_misses = tracker_max_misses.value
    config.tracker_confirm_hits = tracker_confirm_hits.value
    config.pixelcount_threshold = settings.foreground_mask_options.pixelcount_threshold.value
    config.min_area = settings.foreground_mask_options.min_area.value
    config.mog2_history = settings.foreground_mask_options.mog2_history.value
    config.mog2_var_threshold = settings.foreground_mask_options.mog2_var_threshold.value
    config.denoise_kernelsize = settings.foreground_mask_options.denoise_kernelsize.value
    config.mask_transparency = mask_transparency.value
    config.servo_pid_kp = servo_pid_kp.value
    config.servo_pid_ki = servo_pid_ki.value
    config.servo_pid_kd = servo_pid_kd.value
    config.servo_dead_zone = servo_dead_zone.value
    config.servo_kalman_process_noise = servo_kalman_process_noise.value
    config.servo_kalman_meas_noise = servo_kalman_meas_noise.value
    config.save()

    return PikiOptions(
        show_boxes=app_settings.debug_settings.show_boxes,
        show_mask=app_settings.debug_settings.show_mask,
        show_rois=app_settings.debug_settings.show_rois,
        conf_threshold=prob_threshold.value,
        conf_threshold_keep=prob_threshold_keep.value,
        min_consecutive_frames=min_consecutive_frames.value,
        bbox_ema_alpha=bbox_ema_alpha.value,
        ghost_frames_ms=ghost_frames_ms.value,
        tracker_enabled=bool(tracker_enabled.value),
        tracker_iou_threshold=tracker_iou_threshold.value,
        tracker_max_misses=tracker_max_misses.value,
        tracker_confirm_hits=tracker_confirm_hits.value,
        pixelcount_threshold=settings.foreground_mask_options.pixelcount_threshold.value,
        min_area=settings.foreground_mask_options.min_area.value,
        mog2_history=settings.foreground_mask_options.mog2_history.value,
        mog2_var_threshold=settings.foreground_mask_options.mog2_var_threshold.value,
        denoise_kernelsize=settings.foreground_mask_options.denoise_kernelsize.value,
        mask_transparency=mask_transparency.value,
        servo_pid_kp=servo_pid_kp.value,
        servo_pid_ki=servo_pid_ki.value,
        servo_pid_kd=servo_pid_kd.value,
        servo_dead_zone=servo_dead_zone.value,
        servo_kalman_process_noise=servo_kalman_process_noise.value,
        servo_kalman_meas_noise=servo_kalman_meas_noise.value,
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
        show_boxes=app_settings.debug_settings.show_boxes,
        show_mask=app_settings.debug_settings.show_mask,
        show_rois=app_settings.debug_settings.show_rois,
        conf_threshold=prob_threshold.value,
        conf_threshold_keep=prob_threshold_keep.value,
        min_consecutive_frames=min_consecutive_frames.value,
        bbox_ema_alpha=bbox_ema_alpha.value,
        ghost_frames_ms=ghost_frames_ms.value,
        tracker_enabled=bool(tracker_enabled.value),
        tracker_iou_threshold=tracker_iou_threshold.value,
        tracker_max_misses=tracker_max_misses.value,
        tracker_confirm_hits=tracker_confirm_hits.value,
        pixelcount_threshold=settings.foreground_mask_options.pixelcount_threshold.value,
        min_area=settings.foreground_mask_options.min_area.value,
        mog2_history=settings.foreground_mask_options.mog2_history.value,
        mog2_var_threshold=settings.foreground_mask_options.mog2_var_threshold.value,
        denoise_kernelsize=settings.foreground_mask_options.denoise_kernelsize.value,
        mask_transparency=mask_transparency.value,
        servo_pid_kp=servo_pid_kp.value,
        servo_pid_ki=servo_pid_ki.value,
        servo_pid_kd=servo_pid_kd.value,
        servo_dead_zone=servo_dead_zone.value,
        servo_kalman_process_noise=servo_kalman_process_noise.value,
        servo_kalman_meas_noise=servo_kalman_meas_noise.value,
    )


class AimConfigSchema(Schema):
    target_classes: list[str]
    servo_enabled: bool
    target_lock_duration: float
    aim_confidence: float = 0.4
    vertical_angle_offset: float = 0.0
    pan_invert: bool = False
    tilt_invert: bool = False


@api.get("/aim_config", response=AimConfigSchema)
def get_aim_config(request: HttpRequest):
    """Return current servo aim configuration."""
    from .utils.shared import servo_aim_confidence, vertical_angle_offset  # noqa: PLC0415

    return AimConfigSchema(
        target_classes=list(app_settings.aim_settings.target_classes or []),
        servo_enabled=bool(app_settings.aim_settings.servo_enabled),
        target_lock_duration=float(app_settings.aim_settings.target_lock_duration),
        aim_confidence=float(servo_aim_confidence.value),
        vertical_angle_offset=float(vertical_angle_offset.value),
        pan_invert=bool(app_settings.aim_settings.pan_invert),
        tilt_invert=bool(app_settings.aim_settings.tilt_invert),
    )


@api.patch("/aim_config", response=AimConfigSchema)
def update_aim_config(request: HttpRequest, payload: PatchDict[AimConfigSchema]):
    """Update servo aim configuration and persist to database."""
    from .models import AimConfig  # noqa: PLC0415
    from .utils.shared import (  # noqa: PLC0415
        servo_aim_confidence,
        servo_pan_invert,
        servo_tilt_invert,
        vertical_angle_offset,
    )

    config = AimConfig.load()

    if (classes := payload.get("target_classes")) is not None:
        validated = [str(c).strip() for c in classes if str(c).strip() in _YOLO_CLASSES]
        app_settings.aim_settings.target_classes = validated
        config.target_classes = validated

    if (enabled := payload.get("servo_enabled")) is not None:
        app_settings.aim_settings.servo_enabled = bool(enabled)
        config.servo_enabled = bool(enabled)

    if (duration := payload.get("target_lock_duration")) is not None:
        clamped = max(0.0, float(duration))
        app_settings.aim_settings.target_lock_duration = clamped
        config.target_lock_duration = clamped

    if (conf := payload.get("aim_confidence")) is not None:
        clamped = max(0.01, min(1.0, float(conf)))
        app_settings.aim_settings.aim_confidence = clamped
        servo_aim_confidence.value = clamped
        config.aim_confidence = clamped

    if (offset := payload.get("vertical_angle_offset")) is not None:
        clamped = max(-30.0, min(30.0, float(offset)))
        vertical_angle_offset.value = clamped
        config.vertical_angle_offset = clamped

    if (v := payload.get("pan_invert")) is not None:
        app_settings.aim_settings.pan_invert = bool(v)
        servo_pan_invert.value = 1 if v else 0
        config.pan_invert = bool(v)

    if (v := payload.get("tilt_invert")) is not None:
        app_settings.aim_settings.tilt_invert = bool(v)
        servo_tilt_invert.value = 1 if v else 0
        config.tilt_invert = bool(v)

    config.save()

    return AimConfigSchema(
        target_classes=list(app_settings.aim_settings.target_classes or []),
        servo_enabled=bool(app_settings.aim_settings.servo_enabled),
        target_lock_duration=float(app_settings.aim_settings.target_lock_duration),
        aim_confidence=float(servo_aim_confidence.value),
        vertical_angle_offset=float(vertical_angle_offset.value),
        pan_invert=bool(app_settings.aim_settings.pan_invert),
        tilt_invert=bool(app_settings.aim_settings.tilt_invert),
    )


@api.get("/yolo_classes", response=list[str])
def get_yolo_classes(request: HttpRequest):
    """Return the list of all detectable YOLO class names."""
    return _YOLO_CLASSES


class SystemStatus(Schema):
    fps: float


@api.get("/tracker_status", response=SystemStatus)
def get_tracker_status(request: HttpRequest):
    """Return current system status."""
    return SystemStatus(fps=round(fps_counter.fps, 1))


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


# ---------------------------------------------------------------------------
# Video recording / replay endpoints
# ---------------------------------------------------------------------------

from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from django.conf import settings as django_settings  # noqa: E402
from django.http import Http404, HttpResponse  # noqa: E402


class RecordingStatus(Schema):
    is_recording: bool
    elapsed_seconds: float = 0.0
    frame_count: int = 0
    file_path: str = ""
    event_enabled: bool = False
    event_active: bool = False
    event_cooldown_remaining: float = 0.0


class EventRecordingConfigSchema(Schema):
    enabled: bool = False
    pre_buffer_seconds: int = 5
    post_trigger_seconds: int = 10
    trigger_classes: list[str] = []
    cooldown_seconds: int = 30


class EventClipSchema(Schema):
    filename: str
    file: str
    frame_count: int
    time: str
    video_id: int


class VideoInfo(Schema):
    id: int
    filename: str
    url: str
    size_bytes: int
    source: str
    created_at: str
    # Empty for non-event recordings.
    event_id: str = ""
    has_log: bool = False


class ReplayStatus(Schema):
    is_replaying: bool
    video_filename: str = ""
    current_frame: int = 0
    total_frames: int = 0
    video_fps: float = 0.0


@api.post("/recording/start", response={200: RecordingStatus, 409: dict})
def recording_start(request: HttpRequest):
    from .utils.recording import get_recording_stats, is_recording, start_recording  # noqa: PLC0415

    if replaying_active.is_set():
        return 409, {"detail": "Cannot record while replaying."}

    if event_recording_active.is_set():
        return 409, {"detail": "Cannot start manual recording while event recording is active."}

    if is_recording():
        stats = get_recording_stats()
        return RecordingStatus(is_recording=True, **stats)

    videos_dir = Path(django_settings.MEDIA_ROOT) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(videos_dir / f"recording_{timestamp}.mp4")

    # Use actual pipeline FPS so playback speed matches real time.
    actual_fps = fps_counter.fps if fps_counter.fps > 0 else 30.0
    err = start_recording(path, fps=actual_fps)
    if err:
        return 409, {"detail": err}

    recording_active.set()
    stats = get_recording_stats()
    return RecordingStatus(is_recording=True, **stats)


@api.post("/recording/stop", response=RecordingStatus)
def recording_stop(request: HttpRequest):
    from .models import Video  # noqa: PLC0415
    from .utils.recording import get_recording_stats, stop_recording  # noqa: PLC0415

    recording_active.clear()
    path, frame_count, error = stop_recording()

    if path and frame_count > 0:
        file_path = Path(path)
        Video.objects.create(
            filename=file_path.name,
            file=str(file_path.relative_to(django_settings.MEDIA_ROOT)),
            size_bytes=file_path.stat().st_size,
            source="recorded",
        )

    return RecordingStatus(
        is_recording=False,
        elapsed_seconds=0.0,
        frame_count=frame_count,
        file_path=path,
    )


@api.get("/recording/status", response=RecordingStatus)
def recording_status(request: HttpRequest):
    import time  # noqa: PLC0415

    from .utils.recording import get_recording_stats, is_recording  # noqa: PLC0415

    cooldown_remaining = max(0.0, event_recording_cooldown_until - time.time())
    event_fields = {
        "event_enabled": event_recording_enabled.is_set(),
        "event_active": event_recording_active.is_set(),
        "event_cooldown_remaining": round(cooldown_remaining, 1),
    }

    if not is_recording():
        return RecordingStatus(is_recording=False, **event_fields)
    stats = get_recording_stats()
    return RecordingStatus(is_recording=True, **stats, **event_fields)


@api.get("/videos", response=list[VideoInfo])
def videos_list(request: HttpRequest):
    from .models import Video  # noqa: PLC0415

    results = []
    for v in Video.objects.all():
        results.append(
            VideoInfo(
                id=v.pk,
                filename=v.filename,
                url=request.build_absolute_uri(v.file.url),
                size_bytes=v.size_bytes,
                source=v.source,
                created_at=v.created_at.isoformat(),
                event_id=v.event_id or "",
                has_log=bool(v.log_file and v.log_file.name),
            )
        )
    return results


@api.post("/videos/upload", response=VideoInfo)
def videos_upload(request: HttpRequest):
    from .models import Video  # noqa: PLC0415

    uploaded = request.FILES.get("file")
    if not uploaded:
        raise Http404("No file provided")

    video = Video.objects.create(
        filename=uploaded.name,
        size_bytes=uploaded.size,
        source="uploaded",
    )
    video.file.save(uploaded.name, uploaded, save=True)

    return VideoInfo(
        id=video.pk,
        filename=video.filename,
        url=request.build_absolute_uri(video.file.url),
        size_bytes=video.size_bytes,
        source=video.source,
        created_at=video.created_at.isoformat(),
    )


_VIDEO_BIN_RESPONSE = {
    "responses": {
        200: {
            "description": "Video file",
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
            },
        },
    },
}


@api.get("/videos/{video_id}/download", openapi_extra=_VIDEO_BIN_RESPONSE)
def videos_download(request: HttpRequest, video_id: int):
    from django.http.response import FileResponse  # noqa: PLC0415

    from .models import Video  # noqa: PLC0415

    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        raise Http404("Video not found")

    file_path = Path(django_settings.MEDIA_ROOT) / video.file.name
    if not file_path.exists():
        raise Http404("File not found on disk")

    return FileResponse(
        file_path.open("rb"),
        as_attachment=True,
        filename=video.filename,
    )


_LOG_DOWNLOAD_RESPONSE = {
    "responses": {
        200: {
            "description": "Technical-log JSONL sidecar",
            "content": {
                "application/x-ndjson": {"schema": {"type": "string", "format": "binary"}},
            },
        },
    },
}


class EventLogSummary(Schema):
    event_id: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_seconds: float = 0.0
    frames_total: int = 0
    frames_prebuffer: int = 0
    frames_live: int = 0
    detections_total: int = 0
    labels: dict = {}
    splashes: int = 0
    trigger: Optional[dict] = None
    config_snapshot: Optional[dict] = None
    unique_track_ids: list[int] = []


@api.get("/videos/{video_id}/log", openapi_extra=_LOG_DOWNLOAD_RESPONSE)
def videos_log_download(request: HttpRequest, video_id: int):
    from django.http.response import FileResponse  # noqa: PLC0415

    from .models import Video  # noqa: PLC0415

    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        raise Http404("Video not found")
    if not video.log_file or not video.log_file.name:
        raise Http404("This recording has no technical log.")

    file_path = Path(django_settings.MEDIA_ROOT) / video.log_file.name
    if not file_path.exists():
        raise Http404("Log file not found on disk")

    return FileResponse(
        file_path.open("rb"),
        as_attachment=True,
        filename=file_path.name,
        content_type="application/x-ndjson",
    )


@api.get("/videos/{video_id}/log/summary", response=EventLogSummary)
def videos_log_summary(request: HttpRequest, video_id: int):
    from .models import Video  # noqa: PLC0415
    from .utils.event_log import summarize_log  # noqa: PLC0415

    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        raise Http404("Video not found")
    if not video.log_file or not video.log_file.name:
        raise Http404("This recording has no technical log.")

    file_path = Path(django_settings.MEDIA_ROOT) / video.log_file.name
    if not file_path.exists():
        raise Http404("Log file not found on disk")

    summary = summarize_log(str(file_path))
    return EventLogSummary(**summary)


@api.delete("/videos/{video_id}", response={200: dict})
def videos_delete(request: HttpRequest, video_id: int):
    from .models import Video  # noqa: PLC0415

    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        raise Http404("Video not found")

    # Delete the file from disk.
    file_path = Path(django_settings.MEDIA_ROOT) / video.file.name
    if file_path.exists():
        file_path.unlink()

    # Delete the JSONL log sidecar if present.
    if video.log_file and video.log_file.name:
        log_path = Path(django_settings.MEDIA_ROOT) / video.log_file.name
        if log_path.exists():
            log_path.unlink()

    video.delete()
    return {"status": "deleted"}


@api.post("/replay/start/{video_id}", response={200: ReplayStatus, 409: dict})
def replay_start(request: HttpRequest, video_id: int):
    from .models import Video  # noqa: PLC0415
    from .utils.replay import start_replay, is_replaying, get_replay_stats  # noqa: PLC0415

    if recording_active.is_set():
        return 409, {"detail": "Cannot replay while recording."}

    if is_replaying():
        stop_replay()
        replaying_active.clear()

    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        raise Http404("Video not found")

    file_path = Path(django_settings.MEDIA_ROOT) / video.file.name
    if not file_path.exists():
        raise Http404("Video file not found on disk")

    streaming_active.set()
    replaying_active.set()
    start_replay(str(file_path), filename=video.filename)

    import time  # noqa: PLC0415
    time.sleep(0.1)

    stats = get_replay_stats()
    return ReplayStatus(is_replaying=True, **stats)


@api.post("/replay/stop", response={200: dict})
def replay_stop(request: HttpRequest):
    from .utils.replay import stop_replay  # noqa: PLC0415

    stop_replay()
    replaying_active.clear()
    return {"status": "stopped"}


@api.get("/replay/status", response=ReplayStatus)
def replay_status(request: HttpRequest):
    from .utils.replay import get_replay_stats, is_replaying  # noqa: PLC0415

    if not is_replaying():
        return ReplayStatus(is_replaying=False)
    stats = get_replay_stats()
    return ReplayStatus(is_replaying=True, **stats)


# ---------------------------------------------------------------------------
# Event-triggered recording
# ---------------------------------------------------------------------------


@api.get("/event_recording_config", response=EventRecordingConfigSchema)
def get_event_recording_config(request: HttpRequest):
    """Return current event-triggered recording configuration."""
    with event_trigger_classes_lock:
        classes = list(event_trigger_classes)
    return EventRecordingConfigSchema(
        enabled=event_recording_enabled.is_set(),
        pre_buffer_seconds=int(event_pre_buffer_seconds.value),
        post_trigger_seconds=int(event_post_trigger_seconds.value),
        trigger_classes=classes,
        cooldown_seconds=int(event_cooldown_seconds.value),
    )


@api.patch("/event_recording_config", response=EventRecordingConfigSchema)
def update_event_recording_config(request: HttpRequest, payload: PatchDict[EventRecordingConfigSchema]):
    """Update event-triggered recording configuration and persist to DB."""
    from .models import EventRecordingConfig  # noqa: PLC0415
    from .utils.recording import pre_buffer_clear  # noqa: PLC0415

    config = EventRecordingConfig.load()

    if (v := payload.get("enabled")) is not None:
        config.enabled = bool(v)
        if v:
            event_recording_enabled.set()
        else:
            event_recording_enabled.clear()
            pre_buffer_clear()

    if (v := payload.get("pre_buffer_seconds")) is not None:
        clamped = max(1, min(30, int(v)))
        config.pre_buffer_seconds = clamped
        event_pre_buffer_seconds.value = float(clamped)

    if (v := payload.get("post_trigger_seconds")) is not None:
        clamped = max(1, min(60, int(v)))
        config.post_trigger_seconds = clamped
        event_post_trigger_seconds.value = float(clamped)

    if (v := payload.get("trigger_classes")) is not None:
        validated = [str(c).strip() for c in v if str(c).strip() in _YOLO_CLASSES]
        config.trigger_classes = validated
        with event_trigger_classes_lock:
            event_trigger_classes.clear()
            event_trigger_classes.extend([c.lower() for c in validated])

    if (v := payload.get("cooldown_seconds")) is not None:
        clamped = max(0, min(300, int(v)))
        config.cooldown_seconds = clamped
        event_cooldown_seconds.value = float(clamped)

    config.save()

    with event_trigger_classes_lock:
        classes = list(event_trigger_classes)
    return EventRecordingConfigSchema(
        enabled=event_recording_enabled.is_set(),
        pre_buffer_seconds=int(event_pre_buffer_seconds.value),
        post_trigger_seconds=int(event_post_trigger_seconds.value),
        trigger_classes=classes,
        cooldown_seconds=int(event_cooldown_seconds.value),
    )


@api.get("/event_clips", response=list[EventClipSchema])
def get_event_clips(request: HttpRequest):
    """Return recent event-triggered clips and clear the queue.

    The frontend polls this endpoint; each clip is returned only once.
    """
    with event_clip_queue_lock:
        clips = [EventClipSchema(**c) for c in event_clip_queue]
        event_clip_queue.clear()
    return clips


# ---------------------------------------------------------------------------
# Splash (relay/solenoid) configuration
# ---------------------------------------------------------------------------


class SplashConfigSchema(Schema):
    enabled: bool = False
    delay_seconds: float = 0.5
    duration_seconds: float = 1.0
    cooldown_seconds: float = 10.0
    pump_duty: float = 100.0


@api.get("/splash_config", response=SplashConfigSchema)
def get_splash_config(request: HttpRequest):
    """Return current splash (relay/solenoid) configuration."""
    return SplashConfigSchema(
        enabled=splash_enabled.is_set(),
        delay_seconds=float(splash_delay.value),
        duration_seconds=float(splash_duration.value),
        cooldown_seconds=float(splash_cooldown.value),
        pump_duty=float(pump_duty.value),
    )


@api.patch("/splash_config", response=SplashConfigSchema)
def update_splash_config(request: HttpRequest, payload: PatchDict[SplashConfigSchema]):
    """Update splash configuration and persist to database."""
    from .models import SplashConfig  # noqa: PLC0415

    config = SplashConfig.load()

    if (v := payload.get("enabled")) is not None:
        config.enabled = bool(v)
        if v:
            splash_enabled.set()
        else:
            splash_enabled.clear()

    if (v := payload.get("delay_seconds")) is not None:
        clamped = max(0.0, min(30.0, float(v)))
        config.delay_seconds = clamped
        splash_delay.value = clamped

    if (v := payload.get("duration_seconds")) is not None:
        clamped = max(0.01, min(10.0, float(v)))
        config.duration_seconds = clamped
        splash_duration.value = clamped

    if (v := payload.get("cooldown_seconds")) is not None:
        clamped = max(0.0, min(600.0, float(v)))
        config.cooldown_seconds = clamped
        splash_cooldown.value = clamped

    if (v := payload.get("pump_duty")) is not None:
        clamped = max(0.0, min(100.0, float(v)))
        config.pump_duty = clamped
        pump_duty.value = clamped

    config.save()

    return SplashConfigSchema(
        enabled=splash_enabled.is_set(),
        delay_seconds=float(splash_delay.value),
        duration_seconds=float(splash_duration.value),
        cooldown_seconds=float(splash_cooldown.value),
        pump_duty=float(pump_duty.value),
    )


class SplashStatus(Schema):
    state: str = "idle"
    delay_remaining: float = 0.0
    firing_remaining: float = 0.0
    cooldown_remaining: float = 0.0
    enabled: bool = False


@api.get("/splash_status", response=SplashStatus)
def get_splash_status(request: HttpRequest):
    """Return live splash state for the frontend indicator."""
    import time  # noqa: PLC0415

    from .utils import shared as _s  # noqa: PLC0415

    now = time.time()
    is_enabled = _s.splash_enabled.is_set()

    firing_until = _s.splash_firing_until
    if firing_until > 0 and now < firing_until:
        return SplashStatus(
            state="firing",
            firing_remaining=round(firing_until - now, 1),
            enabled=is_enabled,
        )

    armed_at = _s.splash_armed_at
    if armed_at > 0 and is_enabled:
        elapsed = now - armed_at
        remaining = max(0.0, float(_s.splash_delay.value) - elapsed)
        return SplashStatus(
            state="armed",
            delay_remaining=round(remaining, 1),
            enabled=is_enabled,
        )

    cooldown_until = _s.splash_cooldown_until
    if cooldown_until > 0 and now < cooldown_until:
        return SplashStatus(
            state="cooldown",
            cooldown_remaining=round(cooldown_until - now, 1),
            enabled=is_enabled,
        )

    return SplashStatus(state="idle", enabled=is_enabled)


# --------------------------------------------------------------------------- #
# Exclusion zones                                                              #
# --------------------------------------------------------------------------- #


class ExclusionZoneSchema(Schema):
    id: Optional[int] = None
    name: str = "zone"
    enabled: bool = True
    # Normalized [0, 1] (x, y) points.  Must have at least 3.
    points: list[tuple[float, float]]


def _validate_points(points) -> list[list[float]]:
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise Http404("Exclusion zone requires at least 3 points.")
    clean: list[list[float]] = []
    for pt in points:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            raise Http404("Each point must be a [x, y] pair.")
        x = max(0.0, min(1.0, float(pt[0])))
        y = max(0.0, min(1.0, float(pt[1])))
        clean.append([x, y])
    return clean


@api.get("/exclusion_zones", response=list[ExclusionZoneSchema])
def list_exclusion_zones(request: HttpRequest):
    from .models import ExclusionZone  # noqa: PLC0415

    return [
        ExclusionZoneSchema(id=z.id, name=z.name, enabled=z.enabled, points=z.points)
        for z in ExclusionZone.objects.all()
    ]


@api.post("/exclusion_zones", response=ExclusionZoneSchema)
def create_exclusion_zone(request: HttpRequest, payload: ExclusionZoneSchema):
    from .models import ExclusionZone  # noqa: PLC0415
    from .utils import exclusion  # noqa: PLC0415

    zone = ExclusionZone.objects.create(
        name=payload.name or "zone",
        enabled=bool(payload.enabled),
        points=_validate_points(payload.points),
    )
    exclusion.bump_generation()
    return ExclusionZoneSchema(id=zone.id, name=zone.name, enabled=zone.enabled, points=zone.points)


@api.patch("/exclusion_zones/{zone_id}", response=ExclusionZoneSchema)
def update_exclusion_zone(request: HttpRequest, zone_id: int, payload: PatchDict[ExclusionZoneSchema]):
    from .models import ExclusionZone  # noqa: PLC0415
    from .utils import exclusion  # noqa: PLC0415

    try:
        zone = ExclusionZone.objects.get(pk=zone_id)
    except ExclusionZone.DoesNotExist:
        raise Http404("Exclusion zone not found")

    if (v := payload.get("name")) is not None:
        zone.name = str(v)[:64]
    if (v := payload.get("enabled")) is not None:
        zone.enabled = bool(v)
    if (v := payload.get("points")) is not None:
        zone.points = _validate_points(v)
    zone.save()
    exclusion.bump_generation()
    return ExclusionZoneSchema(id=zone.id, name=zone.name, enabled=zone.enabled, points=zone.points)


@api.delete("/exclusion_zones/{zone_id}")
def delete_exclusion_zone(request: HttpRequest, zone_id: int):
    from .models import ExclusionZone  # noqa: PLC0415
    from .utils import exclusion  # noqa: PLC0415

    deleted, _ = ExclusionZone.objects.filter(pk=zone_id).delete()
    if not deleted:
        raise Http404("Exclusion zone not found")
    exclusion.bump_generation()
    return {"ok": True}

