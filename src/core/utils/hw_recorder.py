"""Hardware H.264 recorder for the RDK X5 VPU.

Wraps ``HwH264Encoder`` (libsrcampy-backed VPU) and muxes its annex-B output
into an MP4 container via PyAV. The whole pipeline is NV12 in, MP4 out — no
color conversion on the CPU.

Channel selection: channels 0 (legacy WebRTC) and 1 (WebRTC sub-stream) are
already in use, so recording gets channel 2.
"""

from __future__ import annotations

import io
import logging
import struct
import time
from typing import Any

import av
import numpy as np

logger = logging.getLogger(__name__)

_NAL_SPS = 7
_NAL_PPS = 8
_NAL_IDR = 5

_RECORDING_VPU_CHANNEL = 2


def _build_avcc_extradata(sps: bytes, pps: bytes) -> bytes:
    """Build the AVCDecoderConfigurationRecord (avcC box payload) for MP4."""
    return (
        bytes([1, sps[1], sps[2], sps[3], 0xFF, 0xE1])
        + struct.pack(">H", len(sps)) + sps
        + bytes([0x01])
        + struct.pack(">H", len(pps)) + pps
    )


def _open_template_stream(width: int, height: int, fps: int) -> tuple[Any, Any]:
    """Encode a single black frame with libx264 in memory to seed a template
    Stream — PyAV's ``add_stream_from_template`` is the only reliable way to
    get a fully-configured muxer stream we can then patch extradata on.

    Returns (container, video_stream); the container must outlive the muxing.
    """
    buf = io.BytesIO()
    cont = av.open(buf, "w", format="mp4")
    stream = cont.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "ultrafast"}
    frame = av.VideoFrame.from_ndarray(
        np.zeros((height, width, 3), dtype=np.uint8), format="bgr24",
    )
    for pkt in stream.encode(frame):
        cont.mux(pkt)
    for pkt in stream.encode(None):
        cont.mux(pkt)
    cont.close()

    buf.seek(0)
    src = av.open(buf, "r")
    return src, src.streams.video[0]


class HwH264MP4Writer:
    """Hardware-encoded H.264 writer with a ``cv2.VideoWriter``-like interface.

    Accepts NV12 frames of shape ``(height*3//2, width)``. Raises in
    ``__init__`` if the VPU encoder cannot be opened.
    """

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        from .hw_encoder import HwH264Encoder  # noqa: PLC0415

        self._path = path
        self._fps = max(1, int(round(fps)))
        self._hw = HwH264Encoder(
            channel=_RECORDING_VPU_CHANNEL, width=width, height=height,
        )
        self._aligned_w = self._hw.width
        self._aligned_h = self._hw.height
        self._output: Any = None
        self._stream: Any = None
        self._template_container: Any = None
        self._timescale: int = 0  # set in _init_mp4
        self._start_capture_ns: int | None = None
        self._last_pts: int = -1
        self._closed = False
        logger.info(
            "HwH264MP4Writer opened %s (%dx%d aligned, fps=%d, channel=%d)",
            path, self._aligned_w, self._aligned_h, self._fps,
            _RECORDING_VPU_CHANNEL,
        )

    def isOpened(self) -> bool:  # noqa: N802 (mirrors cv2.VideoWriter API)
        return not self._closed

    def _init_mp4(self, sps: bytes, pps: bytes) -> None:
        self._template_container, template_stream = _open_template_stream(
            self._aligned_w, self._aligned_h, self._fps,
        )
        self._output = av.open(self._path, "w", format="mp4")
        self._stream = self._output.add_stream_from_template(template_stream)
        self._stream.codec_context.extradata = _build_avcc_extradata(sps, pps)
        # The MP4 muxer's timescale comes from the template (typically
        # ``fps * 512``). Compute PTS in those ticks from the real
        # ``capture_ns`` elapsed since the first frame so a sparse pipeline
        # still produces a clip whose playback duration matches wall time.
        self._timescale = int(template_stream.time_base.denominator)

    def write(self, nv12: np.ndarray, capture_ns: int | None = None) -> None:
        """Encode + mux one frame. ``capture_ns`` is the producer-side wall-time
        timestamp (monotonic) — required for the clip's playback timeline to
        match the real capture duration. If omitted, ``time.monotonic_ns()``
        is sampled here (which is only correct if the call is synchronous
        with capture)."""
        if self._closed:
            return
        if capture_ns is None:
            capture_ns = time.monotonic_ns()
        nals = self._hw.encode_nv12(nv12)
        if not nals:
            return

        if self._output is None:
            sps = pps = None
            for nal in nals:
                t = nal[0] & 0x1F
                if t == _NAL_SPS:
                    sps = bytes(nal)
                elif t == _NAL_PPS:
                    pps = bytes(nal)
            if sps is None or pps is None:
                return
            self._init_mp4(sps, pps)

        slice_nals = [n for n in nals if (n[0] & 0x1F) not in (_NAL_SPS, _NAL_PPS)]
        if not slice_nals:
            return

        avcc = b"".join(struct.pack(">I", len(n)) + bytes(n) for n in slice_nals)

        if self._start_capture_ns is None:
            self._start_capture_ns = capture_ns
        pts = (capture_ns - self._start_capture_ns) * self._timescale // 1_000_000_000
        # Guarantee strictly monotonic PTS — the muxer rejects duplicates and
        # nudges by 1 tick anyway, but let's be explicit.
        if pts <= self._last_pts:
            pts = self._last_pts + 1
        self._last_pts = pts

        packet = av.Packet(avcc)
        packet.stream = self._stream
        packet.pts = pts
        packet.dts = pts
        if any((n[0] & 0x1F) == _NAL_IDR for n in slice_nals):
            packet.is_keyframe = True
        self._output.mux_one(packet)

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._output is not None:
            try:
                self._output.close()
            except Exception:
                logger.exception("HwH264MP4Writer: error closing output")
        if self._template_container is not None:
            try:
                self._template_container.close()
            except Exception:
                logger.exception("HwH264MP4Writer: error closing template")
        self._hw.close()


class H264DirectMP4Writer:
    """MP4 muxer for pre-encoded H.264 NAL units — no VPU encode step.

    Accepts pre-encoded NAL units (as ``bytes`` objects, no Annex-B start
    codes) produced by ``HwH264Encoder.encode_nv12()``.  The first call to
    ``write_nals`` that contains SPS and PPS NALs initialises the MP4
    container; subsequent calls mux the slice NALs.  SPS/PPS are written
    into the ``avcC`` extradata box and excluded from packet payloads.
    """

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        self._path = path
        self._fps = max(1, int(round(fps)))
        self._width = width
        self._height = height
        self._output: Any = None
        self._stream: Any = None
        self._template_container: Any = None
        self._timescale: int = 0
        self._start_capture_ns: int | None = None
        self._last_pts: int = -1
        self._closed = False
        logger.info(
            "H264DirectMP4Writer opened %s (%dx%d, fps=%d)",
            path, width, height, self._fps,
        )

    def isOpened(self) -> bool:  # noqa: N802
        return not self._closed

    def _init_mp4(self, sps: bytes, pps: bytes) -> None:
        self._template_container, template_stream = _open_template_stream(
            self._width, self._height, self._fps,
        )
        self._output = av.open(self._path, "w", format="mp4")
        self._stream = self._output.add_stream_from_template(template_stream)
        self._stream.codec_context.extradata = _build_avcc_extradata(sps, pps)
        self._timescale = int(template_stream.time_base.denominator)

    def write_nals(self, nals: list[bytes], capture_ns: int) -> None:
        """Mux one group of pre-encoded NAL units.

        On the first call that includes SPS + PPS the MP4 container is
        initialised.  SPS/PPS are stored in ``avcC`` extradata and excluded
        from packet payloads on all subsequent calls.
        """
        if self._closed or not nals:
            return

        if self._output is None:
            sps = pps = None
            for nal in nals:
                t = nal[0] & 0x1F
                if t == _NAL_SPS:
                    sps = nal
                elif t == _NAL_PPS:
                    pps = nal
            if sps is None or pps is None:
                return
            self._init_mp4(sps, pps)

        slice_nals = [n for n in nals if (n[0] & 0x1F) not in (_NAL_SPS, _NAL_PPS)]
        if not slice_nals:
            return

        avcc = b"".join(struct.pack(">I", len(n)) + n for n in slice_nals)

        if self._start_capture_ns is None:
            self._start_capture_ns = capture_ns
        pts = (capture_ns - self._start_capture_ns) * self._timescale // 1_000_000_000
        if pts <= self._last_pts:
            pts = self._last_pts + 1
        self._last_pts = pts

        packet = av.Packet(avcc)
        packet.stream = self._stream
        packet.pts = pts
        packet.dts = pts
        if any((n[0] & 0x1F) == _NAL_IDR for n in slice_nals):
            packet.is_keyframe = True
        self._output.mux_one(packet)

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._output is not None:
            try:
                self._output.close()
            except Exception:
                logger.exception("H264DirectMP4Writer: error closing output")
        if self._template_container is not None:
            try:
                self._template_container.close()
            except Exception:
                logger.exception("H264DirectMP4Writer: error closing template")
