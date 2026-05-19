import threading
import time
from collections import deque

import numpy as np

from .shared import cv2

_writer: cv2.VideoWriter | None = None
_writer_lock = threading.Lock()
_frame_count = 0
_start_time = 0.0
_output_path: str = ""
_error: str | None = None
_pending_path: str = ""
_pending_fps: float = 30.0

# --- Pre-buffer (circular buffer of recent BGR frames) ---
_pre_buffer: deque[np.ndarray] = deque()
_pre_buffer_lock = threading.Lock()


def pre_buffer_append(frame: np.ndarray, max_seconds: float, fps: float = 30.0) -> None:
    max_frames = int(max_seconds * fps)
    if max_frames <= 0:
        return
    with _pre_buffer_lock:
        _pre_buffer.append(frame.copy())
        while len(_pre_buffer) > max_frames:
            _pre_buffer.popleft()


def pre_buffer_snapshot() -> list[np.ndarray]:
    """Return a snapshot of all buffered frames, oldest first."""
    with _pre_buffer_lock:
        return list(_pre_buffer)


def pre_buffer_clear() -> None:
    with _pre_buffer_lock:
        _pre_buffer.clear()


class EventClipRecorder:
    """Independent VideoWriter for a single event-triggered clip.

    Created with pre-buffer frames flushed first, then receives live frames
    via write_frame().  Uses the same AVC1 / MJPG fallback as manual recording.
    """

    def __init__(self, output_path: str, fps: float,
                 pre_frames: list[np.ndarray],
                 frame_w: int, frame_h: int):
        self._output_path = output_path
        self._frame_count = 0
        self._error: str | None = None

        fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h), isColor=True)
        if not self._writer.isOpened():
            fallback = output_path.rsplit(".", 1)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
            self._writer = cv2.VideoWriter(fallback, fourcc, fps, (frame_w, frame_h), isColor=True)
            if not self._writer.isOpened():
                self._writer = None
                self._error = "Failed to create event VideoWriter"
                return
            self._output_path = fallback

        for frame in pre_frames:
            fh, fw = frame.shape[:2]
            if fw == frame_w and fh == frame_h:
                self._writer.write(frame)
            else:
                self._writer.write(cv2.resize(frame, (frame_w, frame_h)))
            self._frame_count += 1

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer is not None and self._writer.isOpened():
            self._writer.write(frame)
            self._frame_count += 1

    def close(self) -> tuple[str, int, str | None]:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self._output_path, self._frame_count, self._error


def start_recording(output_path: str, fps: float = 30.0) -> str | None:
    """Prepare for recording. The VideoWriter is created on the first write_frame call
    so we can determine frame dimensions from the actual data."""
    global _writer, _frame_count, _start_time, _error, _output_path, _pending_path, _pending_fps
    with _writer_lock:
        if _writer is not None and _writer.isOpened():
            _writer.release()
            _writer = None
        _pending_path = output_path
        _pending_fps = fps
        _frame_count = 0
        _start_time = 0.0
        _output_path = ""
        _error = None
        return None


def _create_writer(frame_w: int, frame_h: int) -> None:
    global _writer, _output_path, _start_time
    fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
    _writer = cv2.VideoWriter(_pending_path, fourcc, _pending_fps, (frame_w, frame_h), isColor=True)
    if not _writer.isOpened():
        fallback_path = _pending_path.rsplit(".", 1)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        _writer = cv2.VideoWriter(fallback_path, fourcc, _pending_fps, (frame_w, frame_h), isColor=True)
        if not _writer.isOpened():
            _writer = None
            return
        _output_path = fallback_path
    else:
        _output_path = _pending_path
    _start_time = time.monotonic()


def write_frame(frame: np.ndarray) -> None:
    global _frame_count, _error
    with _writer_lock:
        if _writer is None and _pending_path:
            h, w = frame.shape[:2]
            _create_writer(w, h)
            if _writer is None:
                _error = "Failed to create VideoWriter"
                return

        if _writer is not None and _writer.isOpened():
            try:
                _writer.write(frame)
                _frame_count += 1
            except Exception as e:
                _error = str(e)
        elif _error is None:
            _error = "VideoWriter not open"


def stop_recording() -> tuple[str, int, str | None]:
    global _writer, _frame_count, _error, _output_path, _pending_path
    with _writer_lock:
        if _writer is not None:
            _writer.release()
            _writer = None
        path = _output_path
        count = _frame_count
        err = _error
        _frame_count = 0
        _output_path = ""
        _pending_path = ""
        _error = None
        return path, count, err


def is_recording() -> bool:
    with _writer_lock:
        return bool(_pending_path) or (_writer is not None and _writer.isOpened())


def get_recording_stats() -> dict:
    with _writer_lock:
        elapsed = time.monotonic() - _start_time if _start_time > 0 else 0.0
        return {
            "elapsed_seconds": round(elapsed, 1),
            "frame_count": _frame_count,
            "file_path": _output_path or _pending_path,
        }
