import asyncio
from typing import Optional

import numpy as np
from django.http import HttpRequest
from django.http.response import StreamingHttpResponse
from ninja import NinjaAPI, PatchDict, Schema

from .utils.shared import (
    app_settings,
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
    is_object_detection_disabled,
    latest_debug_frame,
    latest_frame,
    mask_transparency,
    motion_detector,
    prob_threshold,
    recording_active,
    replaying_active,
    servo_dead_zone,
    servo_pan,
    servo_pid_kd,
    servo_pid_ki,
    servo_pid_kp,
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

    # Persist all current values to DB so they survive restarts.
    config = DetectionConfig.load()
    config.show_boxes = app_settings.debug_settings.show_boxes
    config.show_mask = app_settings.debug_settings.show_mask
    config.show_rois = app_settings.debug_settings.show_rois
    config.conf_threshold = prob_threshold.value
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
    config.save()

    return PikiOptions(
        show_boxes=app_settings.debug_settings.show_boxes,
        show_mask=app_settings.debug_settings.show_mask,
        show_rois=app_settings.debug_settings.show_rois,
        conf_threshold=prob_threshold.value,
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
    )


class AimConfigSchema(Schema):
    target_classes: list[str]
    servo_enabled: bool
    target_lock_duration: float


@api.get("/aim_config", response=AimConfigSchema)
def get_aim_config(request: HttpRequest):
    """Return current servo aim configuration."""
    return AimConfigSchema(
        target_classes=list(app_settings.aim_settings.target_classes or []),
        servo_enabled=bool(app_settings.aim_settings.servo_enabled),
        target_lock_duration=float(app_settings.aim_settings.target_lock_duration),
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

    if (duration := payload.get("target_lock_duration")) is not None:
        clamped = max(0.0, float(duration))
        app_settings.aim_settings.target_lock_duration = clamped
        config.target_lock_duration = clamped

    config.save()

    return AimConfigSchema(
        target_classes=list(app_settings.aim_settings.target_classes or []),
        servo_enabled=bool(app_settings.aim_settings.servo_enabled),
        target_lock_duration=float(app_settings.aim_settings.target_lock_duration),
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


@api.get("/videos/{video_id}/download", response={200: None}, openapi_extra={"responses": {"200": {"content": {"application/octet-stream": {}}, "description": "Video file"}}})
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

