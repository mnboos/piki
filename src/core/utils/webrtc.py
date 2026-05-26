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
import math
from fractions import Fraction
from struct import pack
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
    force_keyframe: bool = False,  # noqa: FBT001, FBT002
) -> tuple[list[bytes], int]:
    """Skip libx264. Use the NALs already attached to the frame."""
    if force_keyframe:
        _s.webrtc_keyframe_requested.set()
        logger.debug("WebRTC keyframe requested by browser (PLI/FIR)")
    nals: list[bytes] = getattr(frame, "_hw_nals", None) or []
    packetized = _h264.H264Encoder._packetize(nals)
    timestamp = _h264.convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
    return packetized, timestamp


_h264.H264Encoder.encode = _passthrough_encode  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# NAL packetizer patches — accept buffer-protocol objects (memoryview)
#
# aiortc's stock packetizers use `bytes + memoryview` which raises TypeError.
# We replace them with versions that only materialise `bytes()` at MTU
# boundaries (~1200 B) rather than copying each whole NAL upfront.  This is
# the main win for large IDR NALs (20–50 KB) coming from hw_encoder.py as
# memoryview slices.
# ---------------------------------------------------------------------------

def _packetize_fu_a_mv(data: bytes | memoryview) -> list[bytes]:  # type: ignore[misc]
    available_size = _h264.PACKET_MAX - _h264.FU_A_HEADER_SIZE
    payload_size = len(data) - _h264.NAL_HEADER_SIZE
    num_packets = math.ceil(payload_size / available_size)
    num_larger_packets = payload_size % num_packets
    package_size = payload_size // num_packets

    f_nri = data[0] & (0x80 | 0x60)
    nal_type = data[0] & 0x1F
    fu_indicator = f_nri | _h264.NAL_TYPE_FU_A

    fu_header_end = bytes([fu_indicator, nal_type | 0x40])
    fu_header_middle = bytes([fu_indicator, nal_type])
    fu_header_start = bytes([fu_indicator, nal_type | 0x80])
    fu_header = fu_header_start

    packages: list[bytes] = []
    offset = _h264.NAL_HEADER_SIZE
    while offset < len(data):
        if num_larger_packets > 0:
            num_larger_packets -= 1
            payload = data[offset : offset + package_size + 1]
            offset += package_size + 1
        else:
            payload = data[offset : offset + package_size]
            offset += package_size
        if offset == len(data):
            fu_header = fu_header_end
        # bytes() here is MTU-sized (~1200 B), not whole-NAL-sized (~20–50 KB).
        packages.append(fu_header + bytes(payload))
        fu_header = fu_header_middle
    return packages


def _packetize_stap_a_mv(  # type: ignore[misc]
    data: bytes | memoryview,
    packages_iterator: object,
) -> tuple[bytes, bytes | memoryview | None]:
    from collections.abc import Iterator  # noqa: PLC0415
    assert isinstance(packages_iterator, Iterator)
    counter = 0
    available_size = _h264.PACKET_MAX - _h264.STAP_A_HEADER_SIZE
    stap_header = _h264.NAL_TYPE_STAP_A | (data[0] & 0xE0)
    payload = bytearray()
    try:
        nalu: bytes | memoryview = data
        while len(nalu) <= available_size and counter < 9:
            stap_header |= nalu[0] & 0x80
            nri = nalu[0] & 0x60
            if stap_header & 0x60 < nri:
                stap_header = stap_header & 0x9F | nri
            available_size -= _h264.LENGTH_FIELD_SIZE + len(nalu)
            counter += 1
            payload += pack("!H", len(nalu)) + bytes(nalu)
            nalu = next(packages_iterator)
        if counter == 0:
            nalu = next(packages_iterator)
    except StopIteration:
        nalu = None
    if counter <= 1:
        return bytes(data), nalu
    return bytes([stap_header]) + bytes(payload), nalu


_h264.H264Encoder._packetize_fu_a = staticmethod(_packetize_fu_a_mv)    # type: ignore[method-assign]
_h264.H264Encoder._packetize_stap_a = staticmethod(_packetize_stap_a_mv)  # type: ignore[method-assign]


def assert_passthrough_active() -> None:
    """Fail loud if a transitive dep upgrade silently undid any patch."""
    if _h264.H264Encoder.encode is not _passthrough_encode:
        raise RuntimeError(
            "aiortc H264 passthrough patch is no longer applied. "
            "An aiortc upgrade may have changed the encode() method.",
        )
    if _h264.H264Encoder._packetize_fu_a is not _packetize_fu_a_mv:
        raise RuntimeError(
            "aiortc _packetize_fu_a patch is no longer applied.",
        )
    if _h264.H264Encoder._packetize_stap_a is not _packetize_stap_a_mv:
        raise RuntimeError(
            "aiortc _packetize_stap_a patch is no longer applied.",
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
