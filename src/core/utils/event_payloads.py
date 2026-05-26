"""Single source of truth for WebSocket event payload shapes.

These dicts mirror today's HTTP response bodies exactly so the frontend
can swap polling for WebSocket subscriptions without changing payload
handling. Producers in `utils/stream.py` and the API publish events via
`core.events.publish`; the consumer's snapshot on connect uses the same
helpers.
"""

import time

from . import shared as _s
from .recording import get_recording_stats, is_recording
from .replay import get_replay_stats, is_replaying


def build_tracker_payload() -> dict:
    """Servo-only payload for the high-frequency ``tracker_status`` WS topic.
    FPS is sent separately on ``pipeline_fps`` at a lower rate so that content
    dedup can suppress unchanged servo positions.
    """
    return {
        "servo": {
            "pan": round(float(_s.servo_pan.value), 2),
            "tilt": round(float(_s.servo_tilt.value), 2),
            "kalman_pan": round(float(_s.servo_kalman_pan.value), 2),
            "kalman_tilt": round(float(_s.servo_kalman_tilt.value), 2),
        },
    }


def build_fps_payload() -> dict:
    return {"fps": round(_s.fps_counter.fps, 1)}


def build_detections_snapshot() -> dict:
    """Last published detections (best-effort, may be empty on startup)."""
    return {
        "frame_ts_ns": 0,
        "detections": [],
    }


def build_rois_payload(rois_normalized: list[list[float]]) -> dict:
    """Wire shape for the `rois` topic. Coordinates are normalized [0,1]."""
    return {"rois": rois_normalized}


def build_rois_snapshot() -> dict:
    return build_rois_payload([])


def build_mask_payload(polygons_normalized: list[list[float]]) -> dict:
    """Wire shape for the `mask` topic.

    Each polygon is a flat ``[x0, y0, x1, y1, ...]`` list of normalized
    coordinates, ready to feed into a Canvas2D path. Only outer contours
    are emitted (no holes) — good enough as a debug visualization.
    """
    return {"polygons": polygons_normalized}


def build_mask_snapshot() -> dict:
    return build_mask_payload([])


def build_splash_payload() -> dict:
    now = time.time()
    is_enabled = _s.splash_enabled.is_set()
    cd = round(float(_s.splash_cooldown.value), 1)

    firing_until = _s.splash_firing_until
    if firing_until > 0 and now < firing_until:
        return {
            "state": "firing",
            "delay_remaining": 0.0,
            "firing_remaining": round(firing_until - now, 1),
            "cooldown_remaining": 0.0,
            "firing_duration": 0.0,
            "cooldown_duration": cd,
            "enabled": is_enabled,
        }

    armed_at = _s.splash_armed_at
    if armed_at > 0 and is_enabled:
        elapsed = now - armed_at
        remaining = max(0.0, float(_s.splash_delay.value) - elapsed)
        return {
            "state": "armed",
            "delay_remaining": round(remaining, 1),
            "firing_remaining": 0.0,
            "cooldown_remaining": 0.0,
            "firing_duration": round(float(_s.splash_duration.value), 1),
            "cooldown_duration": cd,
            "enabled": is_enabled,
        }

    cooldown_until = _s.splash_cooldown_until
    if cooldown_until > 0 and now < cooldown_until:
        return {
            "state": "cooldown",
            "delay_remaining": 0.0,
            "firing_remaining": 0.0,
            "cooldown_remaining": round(cooldown_until - now, 1),
            "firing_duration": 0.0,
            "cooldown_duration": 0.0,
            "enabled": is_enabled,
        }

    return {
        "state": "idle",
        "delay_remaining": 0.0,
        "firing_remaining": 0.0,
        "cooldown_remaining": 0.0,
        "firing_duration": 0.0,
        "cooldown_duration": 0.0,
        "enabled": is_enabled,
    }


def build_recording_payload() -> dict:
    cooldown_remaining = max(0.0, _s.event_recording_cooldown_until - time.time())
    event_fields = {
        "event_enabled": _s.event_recording_enabled.is_set(),
        "event_active": _s.event_recording_active.is_set(),
        "event_cooldown_remaining": round(cooldown_remaining, 1),
    }

    if not is_recording():
        return {
            "is_recording": False,
            "elapsed_seconds": 0.0,
            "frame_count": 0,
            "file_path": "",
            **event_fields,
        }
    stats = get_recording_stats()
    return {"is_recording": True, **stats, **event_fields}


def build_replay_payload() -> dict:
    if not is_replaying():
        return {
            "is_replaying": False,
            "video_filename": "",
            "current_frame": 0,
            "total_frames": 0,
            "video_fps": 0.0,
        }
    stats = get_replay_stats()
    # get_replay_stats() returns {"current_frame", "total_frames", "video_fps", "filename"};
    # the wire field is `video_filename`.
    return {
        "is_replaying": True,
        "video_filename": stats.get("filename", ""),
        "current_frame": stats.get("current_frame", 0),
        "total_frames": stats.get("total_frames", 0),
        "video_fps": stats.get("video_fps", 0.0),
    }


def build_event_clips_snapshot() -> dict:
    """Current accumulated event clips, without draining the queue."""
    with _s.event_clip_queue_lock:
        clips = list(_s.event_clip_queue)
    return {"clips": clips}


def snapshot() -> dict[str, dict]:
    """Full per-topic state for a fresh WS connection."""
    return {
        "tracker_status": build_tracker_payload(),
        "pipeline_fps": build_fps_payload(),
        "splash_status": build_splash_payload(),
        "recording_status": build_recording_payload(),
        "replay_status": build_replay_payload(),
        "event_clips": build_event_clips_snapshot(),
        "detections": build_detections_snapshot(),
        "rois": build_rois_snapshot(),
        "mask": build_mask_snapshot(),
    }
