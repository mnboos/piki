"""WebRTC signaling + pre-encoded H.264 passthrough for aiortc.

aiortc's stock `H264Encoder` builds a libx264 software encoder on the first
frame. We bypass that path entirely: the hardware encoder produces annex-B
NALs (see `hw_encoder.py`), each frame is wrapped in a placeholder
`av.VideoFrame` carrying the NALs as an attribute, and a monkey-patched
`encode()` packetizes them directly into RTP payloads.

The patch is applied at import time and a startup assertion guards against
silent dependency upgrades that would re-introduce the libx264 path.
"""

from __future__ import annotations

import asyncio
import logging
from fractions import Fraction
from typing import TYPE_CHECKING

import av
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)
from aiortc.codecs import h264 as _h264

from . import shared as _s

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# RTP timebase for H.264 is 90 kHz.
VIDEO_TIME_BASE = _h264.VIDEO_TIME_BASE


# ---------------------------------------------------------------------------
# aiortc H264 passthrough patch
# ---------------------------------------------------------------------------

def _passthrough_encode(
    self: _h264.H264Encoder,  # noqa: ARG001
    frame: av.VideoFrame,
    force_keyframe: bool = False,  # noqa: ARG001, FBT001, FBT002
) -> tuple[list[bytes], int]:
    """Skip libx264. Use the NALs already attached to the frame."""
    nals: list[bytes] = getattr(frame, "_hw_nals", None) or []
    packetized = _h264.H264Encoder._packetize(nals)
    timestamp = _h264.convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
    return packetized, timestamp


_h264.H264Encoder.encode = _passthrough_encode  # type: ignore[method-assign]


def assert_passthrough_active() -> None:
    """Fail loud if a transitive dep upgrade silently undid the patch."""
    if _h264.H264Encoder.encode is not _passthrough_encode:
        raise RuntimeError(
            "aiortc H264 passthrough patch is no longer applied. "
            "An aiortc upgrade may have changed the encode() method.",
        )


# ---------------------------------------------------------------------------
# Fanout bus — encoder thread → per-peer asyncio.Queue
# ---------------------------------------------------------------------------

# Each queue belongs to a single RTCPeerConnection. Capacity 4 keeps a tiny
# burst buffer while still dropping in the face of a stalled consumer.
_QUEUE_MAXSIZE = 4

subscribers: set[asyncio.Queue] = set()
_subscribers_lock = asyncio.Lock  # placeholder; we mutate from the loop thread


def _add_subscriber(queue: asyncio.Queue) -> None:
    subscribers.add(queue)
    _s.webrtc_active.set()
    _s.streaming_active.set()
    logger.info("WebRTC subscriber added; total=%d", len(subscribers))


def _remove_subscriber(queue: asyncio.Queue) -> None:
    subscribers.discard(queue)
    if not subscribers:
        _s.webrtc_active.clear()
        _s.streaming_active.clear()
    logger.info("WebRTC subscriber removed; total=%d", len(subscribers))


def _enqueue_drop_oldest(queue: asyncio.Queue, item: tuple[list[bytes], int]) -> None:
    """Push `item`; if the queue is full, drop the oldest entry first."""
    while True:
        try:
            queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return


def webrtc_publish(nals: list[bytes], pts_ns: int) -> None:
    """Fan out one encoded frame to every WebRTC subscriber. Thread-safe.

    Called from the ROS callback (sync) thread; the actual queue mutations
    happen on daphne's event loop via call_soon_threadsafe.
    """
    if not subscribers:
        return
    # Late import to avoid a cycle (events.py imports nothing project-local
    # but we import it lazily anyway in case of import order changes).
    from .. import events as _events  # noqa: PLC0415

    loop = _events._loop  # bound on first WS connect
    if loop is None:
        return
    item = (nals, pts_ns)
    for queue in list(subscribers):
        try:
            loop.call_soon_threadsafe(_enqueue_drop_oldest, queue, item)
        except RuntimeError:
            # Loop closed during shutdown — drop.
            return


# ---------------------------------------------------------------------------
# MediaStreamTrack
# ---------------------------------------------------------------------------

# 1-microsecond timebase so PTS values in ns convert losslessly via integer div.
_PTS_TIME_BASE = Fraction(1, 1_000_000)


class _HwH264Frame(av.VideoFrame):
    """av.VideoFrame subclass that lets us attach the pre-encoded NAL list.

    `av.VideoFrame` is a C extension type with no `__dict__`, so we cannot
    set arbitrary attributes on it. A trivial Python subclass restores that
    ability.
    """


class HwH264Track(MediaStreamTrack):
    kind = "video"

    def __init__(self, queue: asyncio.Queue) -> None:
        super().__init__()
        self._queue = queue

    async def recv(self) -> av.VideoFrame:
        nals, pts_ns = await self._queue.get()
        # The pixel data is ignored by the patched encoder; the 2x2 frame
        # keeps allocations tiny.
        frame = _HwH264Frame(2, 2, format="yuv420p")
        frame.pts = pts_ns // 1000  # ns → µs
        frame.time_base = _PTS_TIME_BASE
        frame._hw_nals = nals
        return frame


# ---------------------------------------------------------------------------
# Signaling — build a PeerConnection from an offer, return the answer
# ---------------------------------------------------------------------------

_peer_connections: set[RTCPeerConnection] = set()


def _h264_only_capabilities() -> list:
    """Return the subset of video capabilities that are H.264."""
    caps = RTCRtpSender.getCapabilities("video")
    h264_codecs = [c for c in caps.codecs if c.mimeType == "video/H264"]
    if not h264_codecs:
        raise RuntimeError("aiortc reports no H.264 codec capabilities")
    return h264_codecs


async def handle_offer(sdp: str, type_: str) -> tuple[str, str]:
    """Build a PC for an incoming offer, return (answer_sdp, answer_type)."""
    pc = RTCPeerConnection()
    _peer_connections.add(pc)

    queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    track = HwH264Track(queue)
    pc.addTrack(track)

    # Pin H.264 baseline; reject Opus + VP8 / VP9 negotiation.
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(_h264_only_capabilities())

    _add_subscriber(queue)

    @pc.on("connectionstatechange")
    async def _on_state_change() -> None:  # noqa: RUF029
        state = pc.connectionState
        logger.info("RTCPeerConnection state=%s", state)
        if state in ("failed", "closed", "disconnected"):
            _remove_subscriber(queue)
            _peer_connections.discard(pc)
            try:
                await pc.close()
            except Exception:  # noqa: BLE001
                logger.exception("error closing peer connection")

    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc.localDescription.sdp, pc.localDescription.type


async def shutdown() -> None:
    """Close every peer connection (called from Django shutdown / tests)."""
    for pc in list(_peer_connections):
        try:
            await pc.close()
        except Exception:  # noqa: BLE001
            logger.exception("error closing peer connection during shutdown")
    _peer_connections.clear()
    subscribers.clear()
    _s.webrtc_active.clear()


# Run the assertion on import so a broken dep upgrade fails fast at boot,
# not silently at first frame.
assert_passthrough_active()
