import asyncio
import logging
import threading
import time
from typing import Optional

import numpy as np
from django.http import HttpRequest
from django.http.response import FileResponse
from ninja import NinjaAPI, PatchDict, Schema

from .models import (
    AimConfig,
    DetectionConfig,
    EventRecordingConfig,
    ExclusionZone,
    SplashConfig,
    Video,
)
from . import events
from .utils import exclusion, sysmetrics, webrtc
from .utils.engine import move_to
from .utils.event_log import summarize_log
from .utils.event_payloads import build_fps_payload, build_recording_payload, build_replay_payload, build_splash_payload, build_tracker_payload
from .utils.recording import (
    NOMINAL_FPS,
    get_recording_stats,
    is_recording,
    start_recording,
    start_recording_h264,
    stop_recording,
)
from .utils.replay import (
    get_replay_stats,
    is_replaying,
    start_replay,
    stop_replay,
)
from .utils.shared import (
    app_settings,
    bbox_ema_alpha,
    coord_ema_alpha,
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
    is_object_detection_disabled,
    min_consecutive_frames,
    motion_detector,
    prob_threshold,
    prob_threshold_keep,
    recording_active,
    replaying_active,
    servo_aim_confidence,
    servo_dead_zone,
    servo_kalman_meas_noise,
    servo_kalman_pan,
    servo_kalman_process_noise,
    servo_kalman_tilt,
    servo_pan,
    servo_pan_invert,
    servo_pid_kd,
    servo_pid_ki,
    servo_pid_kp,
    servo_tilt,
    servo_tilt_invert,
    settings,
    pump_duty,
    splash_cooldown,
    splash_delay,
    splash_duration,
    splash_enabled,
    streaming_active,
    tracker_confirm_hits,
    tracker_delta_t,
    tracker_enabled,
    tracker_inertia,
    tracker_iou_threshold,
    tracker_max_misses,
    vertical_angle_offset,
)
# Splash *state* (splash_armed_at / splash_firing_until / splash_cooldown_until) are
# plain floats reassigned by the inference worker on the live module. Import the
# module itself and read them as attributes — a by-value import would freeze them
# at 0.0 and the status would be stuck on "idle"/"ready" forever.
from .utils import shared as _s

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


class WebRtcOfferSchema(Schema):
    sdp: str
    type: str


class WebRtcAnswerSchema(Schema):
    sdp: str
    type: str


class WebRtcConfigSchema(Schema):
    target_fps: int


@api.post("/webrtc/offer", response=WebRtcAnswerSchema)
async def webrtc_offer(request: HttpRequest, payload: WebRtcOfferSchema):
    """Negotiate a WebRTC peer connection. WHEP-style stateless offer/answer.

    The video track is hardware-encoded H.264 piped from the VPU. Overlays
    (detections, servo crosshair, exclusion zones) are sent separately over
    the /ws/events WebSocket and drawn client-side.
    """
    is_object_detection_disabled.clear()
    sdp, type_ = await webrtc.handle_offer(payload.sdp, payload.type)
    return WebRtcAnswerSchema(sdp=sdp, type=type_)


@api.get("/webrtc/config", response=WebRtcConfigSchema)
def get_webrtc_config(request: HttpRequest):
    """Current WebRTC stream settings."""
    return WebRtcConfigSchema(target_fps=int(_s.webrtc_target_fps.value))


@api.patch("/webrtc/config", response=WebRtcConfigSchema)
def update_webrtc_config(request: HttpRequest, payload: PatchDict[WebRtcConfigSchema]):
    """Update WebRTC stream settings. In-memory only; resets on restart."""
    if (fps := payload.get("target_fps")) is not None:
        _s.webrtc_target_fps.value = max(1, min(60, int(fps)))
    return WebRtcConfigSchema(target_fps=int(_s.webrtc_target_fps.value))


# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------

class CpuMetrics(Schema):
    percent_total: float
    percent_per_core: list[float]
    freq_mhz_per_core: list[float]
    freq_min_mhz: float
    freq_max_mhz: float
    governor: Optional[str]
    core_count: int


class CoolingDevice(Schema):
    cur_state: int
    max_state: int


class MemoryMetrics(Schema):
    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent: float


class SwapMetrics(Schema):
    total_bytes: int
    used_bytes: int
    percent: float


class DiskMetrics(Schema):
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent: float


class NetInterface(Schema):
    name: str
    rx_bytes_per_s: float
    tx_bytes_per_s: float
    signal_dbm: Optional[int] = None


class NetMetrics(Schema):
    rx_bytes_per_s: float
    tx_bytes_per_s: float
    interfaces: list[NetInterface]


class BpuMetrics(Schema):
    load_percent: Optional[int]
    freq_mhz: Optional[float]


class VpuMetrics(Schema):
    clock_mhz: Optional[float]
    load_percent: Optional[float]


class GpuMetrics(Schema):
    freq_mhz: Optional[float]
    load_percent: Optional[float]


class DdrMetrics(Schema):
    freq_mhz: Optional[float]


class IspMetrics(Schema):
    load_percent: Optional[float]


class SystemMetrics(Schema):
    ts: float
    uptime_s: Optional[float]
    load_avg: list[float]
    cpu: CpuMetrics
    memory: MemoryMetrics
    swap: SwapMetrics
    disk_root: DiskMetrics
    net: NetMetrics
    temps_c: dict[str, float]
    bpu: BpuMetrics
    vpu: VpuMetrics
    gpu: GpuMetrics
    ddr: DdrMetrics
    isp: IspMetrics
    cooling: dict[str, CoolingDevice]


@api.get("/metrics", response=SystemMetrics)
def get_metrics(request: HttpRequest):
    """Live system metrics: CPU, memory, temperatures, accelerator state, network."""
    return sysmetrics.collect()


class PikiOptions(Schema):
    show_boxes: bool = True
    conf_threshold: Optional[float] = None
    conf_threshold_keep: Optional[float] = None
    min_consecutive_frames: Optional[int] = None
    bbox_ema_alpha: Optional[float] = None
    coord_ema_alpha: Optional[float] = None
    tracker_enabled: Optional[bool] = None
    tracker_iou_threshold: Optional[float] = None
    tracker_max_misses: Optional[int] = None
    tracker_confirm_hits: Optional[int] = None
    tracker_delta_t: Optional[int] = None
    tracker_inertia: Optional[float] = None
    pixelcount_threshold: Optional[int] = None
    min_area: Optional[int] = None
    mog2_history: Optional[int] = None
    mog2_var_threshold: Optional[int] = None
    denoise_kernelsize: Optional[int] = None
    servo_pid_kp: Optional[float] = None
    servo_pid_ki: Optional[float] = None
    servo_pid_kd: Optional[float] = None
    servo_dead_zone: Optional[float] = None
    servo_kalman_process_noise: Optional[float] = None
    servo_kalman_meas_noise: Optional[float] = None


@api.patch("/update_options", response=PikiOptions)
def update_options(request: HttpRequest, options: PatchDict[PikiOptions]):

    if (v := options.get("show_boxes")) is not None:
        app_settings.debug_settings.show_boxes = v

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

    if (v := options.get("coord_ema_alpha")) is not None:
        coord_ema_alpha.value = max(0.0, min(1.0, float(v)))

    if (v := options.get("tracker_enabled")) is not None:
        tracker_enabled.value = 1 if v else 0

    if (v := options.get("tracker_iou_threshold")) is not None:
        tracker_iou_threshold.value = max(0.0, min(1.0, float(v)))

    if (v := options.get("tracker_max_misses")) is not None:
        tracker_max_misses.value = max(0, int(v))

    if (v := options.get("tracker_confirm_hits")) is not None:
        tracker_confirm_hits.value = max(1, int(v))

    if (v := options.get("tracker_delta_t")) is not None:
        tracker_delta_t.value = max(1, int(v))

    if (v := options.get("tracker_inertia")) is not None:
        tracker_inertia.value = max(0.0, min(1.0, float(v)))

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
    config.conf_threshold = prob_threshold.value
    config.conf_threshold_keep = prob_threshold_keep.value
    config.min_consecutive_frames = min_consecutive_frames.value
    config.bbox_ema_alpha = bbox_ema_alpha.value
    config.coord_ema_alpha = coord_ema_alpha.value
    config.tracker_enabled = bool(tracker_enabled.value)
    config.tracker_iou_threshold = tracker_iou_threshold.value
    config.tracker_max_misses = tracker_max_misses.value
    config.tracker_confirm_hits = tracker_confirm_hits.value
    config.tracker_delta_t = tracker_delta_t.value
    config.tracker_inertia = tracker_inertia.value
    config.pixelcount_threshold = settings.foreground_mask_options.pixelcount_threshold.value
    config.min_area = settings.foreground_mask_options.min_area.value
    config.mog2_history = settings.foreground_mask_options.mog2_history.value
    config.mog2_var_threshold = settings.foreground_mask_options.mog2_var_threshold.value
    config.denoise_kernelsize = settings.foreground_mask_options.denoise_kernelsize.value
    config.servo_pid_kp = servo_pid_kp.value
    config.servo_pid_ki = servo_pid_ki.value
    config.servo_pid_kd = servo_pid_kd.value
    config.servo_dead_zone = servo_dead_zone.value
    config.servo_kalman_process_noise = servo_kalman_process_noise.value
    config.servo_kalman_meas_noise = servo_kalman_meas_noise.value
    config.save()

    return PikiOptions(
        show_boxes=app_settings.debug_settings.show_boxes,
        conf_threshold=prob_threshold.value,
        conf_threshold_keep=prob_threshold_keep.value,
        min_consecutive_frames=min_consecutive_frames.value,
        bbox_ema_alpha=bbox_ema_alpha.value,
        coord_ema_alpha=coord_ema_alpha.value,
        tracker_enabled=bool(tracker_enabled.value),
        tracker_iou_threshold=tracker_iou_threshold.value,
        tracker_max_misses=tracker_max_misses.value,
        tracker_confirm_hits=tracker_confirm_hits.value,
        tracker_delta_t=tracker_delta_t.value,
        tracker_inertia=tracker_inertia.value,
        pixelcount_threshold=settings.foreground_mask_options.pixelcount_threshold.value,
        min_area=settings.foreground_mask_options.min_area.value,
        mog2_history=settings.foreground_mask_options.mog2_history.value,
        mog2_var_threshold=settings.foreground_mask_options.mog2_var_threshold.value,
        denoise_kernelsize=settings.foreground_mask_options.denoise_kernelsize.value,
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
        conf_threshold=prob_threshold.value,
        conf_threshold_keep=prob_threshold_keep.value,
        min_consecutive_frames=min_consecutive_frames.value,
        bbox_ema_alpha=bbox_ema_alpha.value,
        coord_ema_alpha=coord_ema_alpha.value,
        tracker_enabled=bool(tracker_enabled.value),
        tracker_iou_threshold=tracker_iou_threshold.value,
        tracker_max_misses=tracker_max_misses.value,
        tracker_confirm_hits=tracker_confirm_hits.value,
        tracker_delta_t=tracker_delta_t.value,
        tracker_inertia=tracker_inertia.value,
        pixelcount_threshold=settings.foreground_mask_options.pixelcount_threshold.value,
        min_area=settings.foreground_mask_options.min_area.value,
        mog2_history=settings.foreground_mask_options.mog2_history.value,
        mog2_var_threshold=settings.foreground_mask_options.mog2_var_threshold.value,
        denoise_kernelsize=settings.foreground_mask_options.denoise_kernelsize.value,
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


class ServoStateSchema(Schema):
    pan: float
    tilt: float
    kalman_pan: float
    kalman_tilt: float


class SystemStatus(Schema):
    fps: float
    servo: ServoStateSchema


@api.get("/tracker_status", response=SystemStatus)
def get_tracker_status(request: HttpRequest):
    """Return current system status (FPS + servo position)."""
    payload = {**build_tracker_payload(), **build_fps_payload()}
    return SystemStatus(**payload)


class ServoMoveSchema(Schema):
    pan_angle: float   # degrees, -90 (left) … +90 (right)
    tilt_angle: float  # degrees, -90 (up)   … +90 (down)


class ServoPositionSchema(Schema):
    pan_angle: float
    tilt_angle: float


@api.post("/servo/move", response=ServoPositionSchema)
def servo_move(request: HttpRequest, payload: ServoMoveSchema):
    """Manually command both servos to explicit angles (debug / calibration mode)."""

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

    # Camera's nominal rate; the measured rate can dip below this during
    # bursts and would otherwise cause slow-motion playback.
    fps = float(NOMINAL_FPS)

    from .utils import stream as _stream  # noqa: PLC0415
    dims = _stream.get_rolling_buffer_dims()
    if dims is not None:
        # Rolling buffer is active — tap its output directly instead of
        # spinning up a second on-demand VPU encode on ch2.
        err = start_recording_h264(path, fps=fps, frame_w=dims[0], frame_h=dims[1])
    else:
        err = start_recording(path, fps=fps)
    if err:
        return 409, {"detail": err}

    recording_active.set()
    events.publish("recording_status", build_recording_payload())
    stats = get_recording_stats()
    return RecordingStatus(is_recording=True, **stats)


@api.post("/recording/stop", response=RecordingStatus)
def recording_stop(request: HttpRequest):

    recording_active.clear()
    path, frame_count, error = stop_recording()
    events.publish("recording_status", build_recording_payload())

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


_VIDEOS_CACHE_TTL_S = 10.0
_videos_cache_lock = threading.Lock()
# Keyed by (scheme, host) so absolute URLs stay valid across hosts.
_videos_cache: dict[tuple[str, str], tuple[float, list[VideoInfo]]] = {}


@api.get("/videos", response=list[VideoInfo])
def videos_list(request: HttpRequest):
    key = (request.scheme, request.get_host())
    now = time.monotonic()
    with _videos_cache_lock:
        entry = _videos_cache.get(key)
        if entry is not None and entry[0] > now:
            return entry[1]

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

    with _videos_cache_lock:
        _videos_cache[key] = (now + _VIDEOS_CACHE_TTL_S, results)
    return results


@api.post("/videos/upload", response=VideoInfo)
def videos_upload(request: HttpRequest):

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

    time.sleep(0.1)

    events.publish("replay_status", build_replay_payload())
    stats = get_replay_stats()
    return ReplayStatus(is_replaying=True, **stats)


@api.post("/replay/stop", response={200: dict})
def replay_stop(request: HttpRequest):

    stop_replay()
    replaying_active.clear()
    events.publish("replay_status", build_replay_payload())
    return {"status": "stopped"}


@api.get("/replay/status", response=ReplayStatus)
def replay_status(request: HttpRequest):

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

    config = EventRecordingConfig.load()

    if (v := payload.get("enabled")) is not None:
        config.enabled = bool(v)
        if v:
            event_recording_enabled.set()
        else:
            event_recording_enabled.clear()
            from .utils import stream as _stream  # noqa: PLC0415
            _stream.destroy_rolling_buffer()

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


class SplashActivateResponse(Schema):
    ok: bool = True
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
    firing_duration: float = 0.0
    cooldown_duration: float = 0.0
    enabled: bool = False


@api.get("/splash_status", response=SplashStatus)
def get_splash_status(request: HttpRequest):
    """Return live splash state for the frontend indicator."""


    now = time.time()
    is_enabled = splash_enabled.is_set()
    cd = round(float(splash_cooldown.value), 1)

    firing_until = _s.splash_firing_until
    if firing_until > 0 and now < firing_until:
        return SplashStatus(
            state="firing",
            firing_remaining=round(firing_until - now, 1),
            cooldown_duration=cd,
            enabled=is_enabled,
        )

    armed_at = _s.splash_armed_at
    if armed_at > 0 and is_enabled:
        elapsed = now - armed_at
        remaining = max(0.0, float(splash_delay.value) - elapsed)
        return SplashStatus(
            state="armed",
            delay_remaining=round(remaining, 1),
            firing_duration=round(float(splash_duration.value), 1),
            cooldown_duration=cd,
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


@api.post("/splash/activate", response=SplashActivateResponse)
def activate_splash(request: HttpRequest):
    """Manually fire the pump after the configured delay, then cooldown."""
    from .utils.engine import activate_pump  # noqa: PLC0415

    delay = float(splash_delay.value)
    duration = float(splash_duration.value)
    duty = float(pump_duty.value)
    cooldown = float(splash_cooldown.value)

    now = time.time()
    _s.splash_armed_at = now
    # Prevent the inference loop from interfering during the full sequence.
    _s.splash_cooldown_until = now + delay + duration + cooldown

    events.publish("splash_status", build_splash_payload())

    def _fire_after_delay():
        time.sleep(delay)
        _s.splash_armed_at = 0.0
        _s.splash_firing_until = time.time() + duration
        activate_pump(duration, duty)
        events.publish("splash_status", build_splash_payload())

    threading.Thread(target=_fire_after_delay, daemon=True, name="splash-manual").start()
    return SplashActivateResponse(ok=True)


# --------------------------------------------------------------------------- #
# Gamepad endpoints                                                             #
# --------------------------------------------------------------------------- #


class GamepadStatusSchema(Schema):
    connected: bool
    enabled: bool
    pan: float
    tilt: float


@api.get("/gamepad/status", response=GamepadStatusSchema)
def get_gamepad_status(request: HttpRequest):
    from .utils.gamepad import gamepad_connected, gamepad_enabled, gamepad_pan, gamepad_tilt, _gamepad_state_lock  # noqa: PLC0415
    with _gamepad_state_lock:
        return GamepadStatusSchema(
            connected=gamepad_connected.is_set(),
            enabled=gamepad_enabled.is_set(),
            pan=round(gamepad_pan, 1),
            tilt=round(gamepad_tilt, 1),
        )


class GamepadEnableSchema(Schema):
    enabled: bool


@api.post("/gamepad/enable", response=GamepadStatusSchema)
def gamepad_enable_endpoint(request: HttpRequest, payload: GamepadEnableSchema):
    from .utils.gamepad import gamepad_connected, gamepad_enabled, gamepad_pan, gamepad_tilt, _gamepad_state_lock  # noqa: PLC0415
    if payload.enabled:
        gamepad_enabled.set()
    else:
        gamepad_enabled.clear()
    with _gamepad_state_lock:
        return GamepadStatusSchema(
            connected=gamepad_connected.is_set(),
            enabled=gamepad_enabled.is_set(),
            pan=round(gamepad_pan, 1),
            tilt=round(gamepad_tilt, 1),
        )


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

    return [
        ExclusionZoneSchema(id=z.id, name=z.name, enabled=z.enabled, points=z.points)
        for z in ExclusionZone.objects.all()
    ]


@api.post("/exclusion_zones", response=ExclusionZoneSchema)
def create_exclusion_zone(request: HttpRequest, payload: ExclusionZoneSchema):

    zone = ExclusionZone.objects.create(
        name=payload.name or "zone",
        enabled=bool(payload.enabled),
        points=_validate_points(payload.points),
    )
    exclusion.bump_generation()
    return ExclusionZoneSchema(id=zone.id, name=zone.name, enabled=zone.enabled, points=zone.points)


@api.patch("/exclusion_zones/{zone_id}", response=ExclusionZoneSchema)
def update_exclusion_zone(request: HttpRequest, zone_id: int, payload: PatchDict[ExclusionZoneSchema]):

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

    deleted, _ = ExclusionZone.objects.filter(pk=zone_id).delete()
    if not deleted:
        raise Http404("Exclusion zone not found")
    exclusion.bump_generation()
    return {"ok": True}

