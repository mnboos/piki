import threading
import time

import numpy as np

from .. import events
from .shared import cv2, preview_downscale_factor

_stop_event = threading.Event()
_thread: threading.Thread | None = None
_stats: dict = {
    "current_frame": 0,
    "total_frames": 0,
    "video_fps": 0.0,
    "filename": "",
}


def _build_nv12_from_gray(gray: np.ndarray) -> np.ndarray:
    """Build a synthetic NV12 frame from a decimated grayscale image.

    Upscales the grayscale (which matches frame_lores) back to full resolution,
    then appends a neutral UV plane (128 = no chroma).
    """
    step = preview_downscale_factor
    h, w = gray.shape
    y_plane = cv2.resize(gray, (w * step, h * step), interpolation=cv2.INTER_NEAREST)
    uv_h = y_plane.shape[0] // 2
    uv_w = y_plane.shape[1]
    uv_plane = np.full((uv_h, uv_w), 128, dtype=np.uint8)
    return np.vstack([y_plane, uv_plane])


def _replay_loop(video_path: str):
    global _stats
    from .stream import process_frame  # noqa: PLC0415

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    frame_delay = 1.0 / video_fps

    _stats["total_frames"] = total_frames
    _stats["video_fps"] = round(video_fps, 1)

    frame_idx = 0
    next_frame_time = time.monotonic()

    while not _stop_event.is_set():
        ret, bgr = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            next_frame_time = time.monotonic()
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

        # Feed through the exact same pipeline as the live camera.
        # process_frame handles motion detection, AI inference, aiming,
        # and display — the pipeline never knows the source is a replay.
        nv12 = _build_nv12_from_gray(gray)
        frame_h = nv12.shape[0] * 2 // 3
        process_frame(nv12_frame=nv12, frame_h=frame_h)

        frame_idx += 1
        _stats["current_frame"] = frame_idx

        from .event_payloads import build_replay_payload  # noqa: PLC0415
        events.publish_throttled("replay_status", build_replay_payload(), 0.5)

        sleep_time = next_frame_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        next_frame_time += frame_delay

    cap.release()


def start_replay(video_path: str, filename: str = "") -> None:
    global _thread, _stop_event, _stats
    _stop_event.clear()
    _stats = {"current_frame": 0, "total_frames": 0, "video_fps": 0.0, "filename": filename}

    if _thread is not None and _thread.is_alive():
        _stop_event.set()
        _thread.join(timeout=2.0)

    _thread = threading.Thread(target=_replay_loop, args=(video_path,), daemon=True)
    _thread.start()


def stop_replay() -> None:
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=2.0)


def is_replaying() -> bool:
    return _thread is not None and _thread.is_alive()


def get_replay_stats() -> dict:
    return dict(_stats)
