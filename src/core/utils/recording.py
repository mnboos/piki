import logging
import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np

try:
    import piki_nal as _nal
    _HAS_PIKI_NAL = True
except ImportError:
    _nal = None  # type: ignore[assignment]
    _HAS_PIKI_NAL = False

logger = logging.getLogger(__name__)

# Camera's nominal capture rate. Used as the recording timestamp rate and the
# rolling buffer's wall-time-to-frames conversion ceiling. Must match
# `mipi_image_framerate` in run.sh (currently 30 fps).
NOMINAL_FPS = 30

# NAL unit type constants (lower 5 bits of the first byte).
_NAL_IDR = 5


def _open_writer(path: str, fps: float, frame_w: int, frame_h: int):
    """Open a hardware-encoded H.264 → MP4 writer wrapped in an async queue."""
    from .hw_recorder import HwH264MP4Writer  # noqa: PLC0415
    sync = HwH264MP4Writer(path, fps, frame_w, frame_h)
    return _AsyncWriter(sync), path


class _AsyncWriter:
    """Run a synchronous frame writer on a dedicated daemon thread so the
    camera callback never blocks on encode I/O. Drops the oldest queued frame
    on overflow so the producer is never throttled.

    Use ``write_sync`` for one-off bursts that must not be dropped (e.g. the
    pre-buffer flush at recording start); call ``start_async`` once before
    switching to ``write``."""

    _QUEUE_SIZE = 8

    def __init__(self, sync_writer: Any) -> None:
        self._sync = sync_writer
        self._sync_lock = threading.Lock()
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self._QUEUE_SIZE)
        self._dropped = 0
        self._stopping = False
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame, capture_ns = item
            with self._sync_lock:
                if self._stopping:
                    return
                try:
                    self._sync.write(frame, capture_ns=capture_ns)
                except Exception:
                    logger.exception("Async recording writer failed")

    def write_sync(self, frame: np.ndarray, capture_ns: int) -> None:
        """Write directly to the underlying writer, blocking until done.

        Safe to call before ``start_async``; afterwards it serializes against
        the worker thread via an internal lock.
        """
        with self._sync_lock:
            if self._stopping:
                return
            self._sync.write(frame, capture_ns=capture_ns)

    def start_async(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="recording-writer", daemon=True,
        )
        self._thread.start()

    def isOpened(self) -> bool:  # noqa: N802
        return not self._stopping and self._sync.isOpened()

    def write(self, frame: np.ndarray, capture_ns: int) -> None:
        if self._stopping:
            return
        if self._thread is None:
            # No worker yet — fall back to a synchronous write so frames aren't
            # silently dropped before ``start_async`` is called.
            self.write_sync(frame, capture_ns)
            return
        item = (frame, capture_ns)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
                self._dropped += 1
            except queue.Empty:
                pass

    def release(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=5.0)
        with self._sync_lock:
            self._sync.release()
        if self._dropped:
            logger.warning(
                "Async recording writer dropped %d frame(s) due to encoder backpressure",
                self._dropped,
            )


class _AsyncNalWriter:
    """Run ``H264DirectMP4Writer.write_nals`` on a dedicated daemon thread.

    Mirrors ``_AsyncWriter`` but the queue carries ``(nals, capture_ns)``
    tuples of pre-encoded H.264 NAL bytes instead of NV12 frames.  Drops the
    oldest entry on overflow so the camera callback is never throttled.
    """

    _QUEUE_SIZE = 8

    def __init__(self, sync_writer: Any) -> None:
        self._sync = sync_writer
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self._QUEUE_SIZE)
        self._dropped = 0
        self._stopping = False
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            nals, capture_ns = item
            if not self._stopping:
                try:
                    self._sync.write_nals(nals, capture_ns)
                except Exception:
                    logger.exception("Async NAL writer failed")

    def start_async(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="recording-nal-writer", daemon=True,
        )
        self._thread.start()

    def isOpened(self) -> bool:  # noqa: N802
        return not self._stopping and self._sync.isOpened()

    def write(self, nals: list, capture_ns: int) -> None:
        if self._stopping:
            return
        item = (nals, capture_ns)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
                self._dropped += 1
            except queue.Empty:
                pass

    def write_nals(self, nals: list, capture_ns: int) -> None:
        """Alias for ``write`` — API-compatible with ``RustMp4Writer``."""
        self.write(nals, capture_ns)

    def release(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=5.0)
        self._sync.release()
        if self._dropped:
            logger.warning(
                "Async NAL writer dropped %d frame(s) due to muxer backpressure",
                self._dropped,
            )


class H264RollingBuffer:
    """Always-on H.264 pre-buffer backed by a VPU encoder on a dedicated channel.

    Encodes every NV12 frame and stores the resulting NAL units in a ring deque
    pruned by wall time and a hard frame cap.  ``snapshot()`` returns an
    IDR-aligned window so the extracted segment is always independently
    decodable without any prior reference frames.

    Not thread-safe for the encode path (``push`` must be called from a single
    producer thread).  ``snapshot`` acquires the ring lock independently.
    """

    def __init__(self, channel: int, width: int, height: int) -> None:
        from .hw_encoder import HwH264Encoder  # noqa: PLC0415

        self._enc = HwH264Encoder(channel=channel, width=width, height=height)
        self._ring: deque[tuple[int, list, bool]] = deque()
        self._lock = threading.Lock()

    @property
    def width(self) -> int:
        return self._enc.width

    @property
    def height(self) -> int:
        return self._enc.height

    def push(self, nv12: np.ndarray, capture_ns: int, max_seconds: float) -> list:
        """Encode one NV12 frame, store its NALs in the ring, and return them.

        The returned list is the same object stored in the ring, so callers
        can use it directly without an extra copy.  When ``piki_nal`` is
        available the list contains ``NalSlice`` handles (zero-copy after the
        one-time VPU-output copy); otherwise plain ``bytes`` objects.
        """
        nals_mv = self._enc.encode_nv12(nv12)
        if not nals_mv:
            return []
        if _HAS_PIKI_NAL:
            nals = _nal.slices_from_views(nals_mv)
            is_idr = any(s.is_idr for s in nals)
        else:
            # Fallback: copy memoryviews → bytes (encoder reuses its buffer).
            nals = [bytes(mv) for mv in nals_mv]
            is_idr = any((n[0] & 0x1F) == _NAL_IDR for n in nals)

        cutoff_ns = capture_ns - int(max_seconds * 1_000_000_000)
        max_frames = int(max_seconds * NOMINAL_FPS) + NOMINAL_FPS
        with self._lock:
            self._ring.append((capture_ns, nals, is_idr))
            while self._ring and self._ring[0][0] < cutoff_ns:
                self._ring.popleft()
            while len(self._ring) > max_frames:
                self._ring.popleft()
        return nals

    def snapshot(self, seconds: float) -> list[tuple[int, list]]:
        """Return an IDR-aligned window covering the last ``seconds`` seconds.

        Walks backwards from the first frame at or after the cutoff timestamp
        to find the most recent IDR frame.  This guarantees the returned
        segment can be decoded independently without any prior reference frame.
        If no IDR exists before the desired window, the first available IDR in
        the ring is used as the start point.
        """
        cutoff_ns = time.monotonic_ns() - int(seconds * 1_000_000_000)
        with self._lock:
            frames: list[tuple[int, list[bytes], bool]] = list(self._ring)

        if not frames:
            return []

        # Find index of first frame at or after the cutoff (desired window start).
        window_start = len(frames)
        for i, (ts, _, _) in enumerate(frames):
            if ts >= cutoff_ns:
                window_start = i
                break

        # Walk backwards to find the last IDR at or before window_start.
        idr_idx: int | None = None
        for i in range(min(window_start, len(frames) - 1), -1, -1):
            if frames[i][2]:  # is_idr
                idr_idx = i
                break

        if idr_idx is None:
            # No IDR before the desired window — use the first IDR in the ring.
            for i in range(len(frames)):
                if frames[i][2]:
                    idr_idx = i
                    break

        if idr_idx is None:
            return []

        return [(ts, nals) for ts, nals, _ in frames[idr_idx:]]

    def close(self) -> None:
        self._enc.close()
        with self._lock:
            self._ring.clear()


# --- Manual recording (module-level state) ---
_writer: Any = None
_writer_lock = threading.Lock()
_frame_count = 0
_start_time = 0.0
_output_path: str = ""
_error: str | None = None
# NV12 on-demand path (used when rolling buffer is not active)
_pending_path: str = ""
_pending_fps: float = float(NOMINAL_FPS)
# H.264 direct path (used when rolling buffer is active)
_pending_h264_path: str = ""
_pending_h264_fps: float = float(NOMINAL_FPS)
_pending_h264_w: int = 0
_pending_h264_h: int = 0


class EventClipRecorder:
    """Independent writer for a single event-triggered H.264 clip.

    Created with a list of pre-encoded H.264 NAL groups (from the rolling
    buffer), then receives live NALs via ``write_nals()``.  Uses
    ``H264DirectMP4Writer`` — no VPU re-encode at clip start.
    """

    def __init__(self, output_path: str, fps: float,
                 pre_frames: list[tuple[int, list]],
                 frame_w: int, frame_h: int):
        self._output_path = output_path
        self._frame_count = 0
        self._error: str | None = None
        self._async_writer: Any = None

        if _HAS_PIKI_NAL:
            try:
                writer = _nal.RustMp4Writer(output_path, fps, frame_w, frame_h)
            except Exception as exc:
                self._error = f"Failed to create event recorder: {exc}"
                logger.exception("EventClipRecorder open failed")
                return

            for capture_ns, nals in pre_frames:
                try:
                    writer.write_nals(nals, capture_ns)
                    self._frame_count += 1
                except Exception:
                    logger.exception("EventClipRecorder: error writing pre-frame")

            self._async_writer = writer
        else:
            from .hw_recorder import H264DirectMP4Writer  # noqa: PLC0415

            try:
                direct = H264DirectMP4Writer(output_path, fps, frame_w, frame_h)
            except Exception as exc:
                self._error = f"Failed to create event recorder: {exc}"
                logger.exception("EventClipRecorder open failed")
                return

            # Mux pre-buffer H.264 frames synchronously — muxing is fast (no
            # VPU encode) so no overflow risk.  Then switch to async for live.
            for capture_ns, nals in pre_frames:
                try:
                    direct.write_nals(nals, capture_ns)
                    self._frame_count += 1
                except Exception:
                    logger.exception("EventClipRecorder: error writing pre-frame")

            self._async_writer = _AsyncNalWriter(direct)
            self._async_writer.start_async()

    def write_nals(self, nals: list, capture_ns: int) -> None:
        if self._async_writer is not None and self._async_writer.isOpened():
            self._async_writer.write_nals(nals, capture_ns)
            self._frame_count += 1

    def close(self) -> tuple[str, int, str | None]:
        if self._async_writer is not None:
            self._async_writer.release()
            self._async_writer = None
        return self._output_path, self._frame_count, self._error


def start_recording(output_path: str, fps: float = float(NOMINAL_FPS)) -> str | None:
    """Prepare for NV12 on-demand recording (rolling buffer not active).

    The ``HwH264MP4Writer`` is created lazily on the first ``write_frame``
    call so frame dimensions are known from real data.
    """
    global _writer, _frame_count, _start_time, _error, _output_path
    global _pending_path, _pending_fps, _pending_h264_path
    with _writer_lock:
        if _writer is not None and _writer.isOpened():
            _writer.release()
            _writer = None
        _pending_path = output_path
        _pending_fps = fps
        _pending_h264_path = ""
        _frame_count = 0
        _start_time = 0.0
        _output_path = ""
        _error = None
        return None


def start_recording_h264(output_path: str, fps: float,
                         frame_w: int, frame_h: int) -> str | None:
    """Prepare for direct H.264 recording (rolling buffer active).

    The ``H264DirectMP4Writer`` is created lazily on the first ``write_nals``
    call.  Callers must know frame dimensions upfront (from the rolling
    buffer encoder's aligned dimensions).
    """
    global _writer, _frame_count, _start_time, _error, _output_path
    global _pending_h264_path, _pending_h264_fps, _pending_h264_w, _pending_h264_h, _pending_path
    with _writer_lock:
        if _writer is not None and _writer.isOpened():
            _writer.release()
            _writer = None
        _pending_h264_path = output_path
        _pending_h264_fps = fps
        _pending_h264_w = frame_w
        _pending_h264_h = frame_h
        _pending_path = ""
        _frame_count = 0
        _start_time = 0.0
        _output_path = ""
        _error = None
        return None


def _create_writer(frame_w: int, frame_h: int) -> None:
    global _writer, _output_path, _start_time
    try:
        _writer, _output_path = _open_writer(_pending_path, _pending_fps, frame_w, frame_h)
    except Exception:
        logger.exception("Failed to open recording writer")
        _writer = None
        return
    # Manual recording has no pre-buffer flush; go straight to async so the
    # camera callback never blocks on the encoder.
    _writer.start_async()
    _start_time = time.monotonic()


def _create_h264_writer() -> None:
    global _writer, _output_path, _start_time
    if _HAS_PIKI_NAL:
        try:
            _writer = _nal.RustMp4Writer(
                _pending_h264_path, _pending_h264_fps, _pending_h264_w, _pending_h264_h,
            )
            _output_path = _pending_h264_path
        except Exception:
            logger.exception("Failed to open RustMp4Writer")
            _writer = None
            return
    else:
        from .hw_recorder import H264DirectMP4Writer  # noqa: PLC0415
        try:
            direct = H264DirectMP4Writer(
                _pending_h264_path, _pending_h264_fps, _pending_h264_w, _pending_h264_h,
            )
            _writer = _AsyncNalWriter(direct)
            _output_path = _pending_h264_path
        except Exception:
            logger.exception("Failed to open H264DirectMP4Writer")
            _writer = None
            return
        _writer.start_async()
    _start_time = time.monotonic()


def write_frame(frame: np.ndarray) -> None:
    """Write an NV12 frame to the active on-demand recording."""
    global _frame_count, _error
    with _writer_lock:
        if _writer is None and _pending_path:
            # NV12 shape is (H*3//2, W); recover the image dimensions.
            h = frame.shape[0] * 2 // 3
            w = frame.shape[1]
            _create_writer(w, h)
            if _writer is None:
                _error = "Failed to create VideoWriter"
                return

        if _writer is not None and _writer.isOpened():
            try:
                _writer.write(frame, capture_ns=time.monotonic_ns())
                _frame_count += 1
            except Exception as e:
                _error = str(e)
        elif _error is None:
            _error = "VideoWriter not open"


def write_nals(nals: list, capture_ns: int) -> None:
    """Write a group of pre-encoded NAL units to the active H.264 recording."""
    global _frame_count, _error
    if not nals:
        return
    with _writer_lock:
        if _writer is None and _pending_h264_path:
            _create_h264_writer()
            if _writer is None:
                _error = "Failed to create H264 writer"
                return

        if _writer is not None and _writer.isOpened():
            try:
                _writer.write_nals(nals, capture_ns)
                _frame_count += 1
            except Exception as e:
                _error = str(e)
        elif _error is None:
            _error = "H264 writer not open"


def stop_recording() -> tuple[str, int, str | None]:
    global _writer, _frame_count, _error, _output_path, _pending_path, _pending_h264_path
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
        _pending_h264_path = ""
        _error = None
        return path, count, err


def is_recording() -> bool:
    with _writer_lock:
        return (
            bool(_pending_path) or bool(_pending_h264_path)
            or (_writer is not None and _writer.isOpened())
        )


def get_recording_stats() -> dict:
    with _writer_lock:
        elapsed = time.monotonic() - _start_time if _start_time > 0 else 0.0
        return {
            "elapsed_seconds": round(elapsed, 1),
            "frame_count": _frame_count,
            "file_path": _output_path or _pending_path or _pending_h264_path,
        }
