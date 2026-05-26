"""Hardware H.264 encoder wrapper for the RDK X5 VPU.

The Horizon ``libsrcampy.Encoder`` exposes a thin Python binding around the
``spdev::VPPEncode`` C++ class. We wrap it to:

* Pull NV12 frames out of the ROS callback and into annex-B H.264 NALs.
* Cache SPS/PPS and re-inject them in front of every IDR so a fresh
  ``RTCPeerConnection`` can decode without waiting for the next stream
  restart. (The hardware encoder emits SPS+PPS only on its very first
  output buffer.)

The codec runs on the VPU so this call should not consume meaningful CPU.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# Codec type passed as the 2nd argument to libsrcampy.Encoder.encode().
# Per rdk-samples/05_web_display_camera_sample/mipi_camera_web.py and the
# Sunrise multimedia samples: 1=H264, 2=H265, 3=MJPEG.
_TYPE_H264 = 1

# NAL unit type values (lower 5 bits of nal_unit_header byte).
_NAL_TYPE_IDR = 5
_NAL_TYPE_SPS = 7
_NAL_TYPE_PPS = 8


def _split_annex_b(buf: bytes) -> list[memoryview]:
    """Split an annex-B H.264 bitstream into NAL units (no start codes).

    Returns zero-copy ``memoryview`` slices over *buf* rather than new
    ``bytes`` objects.  The caller keeps *buf* alive for as long as the
    returned views are needed.
    """
    n = len(buf)
    starts: list[tuple[int, int]] = []
    i = 0
    while i < n - 2:
        if buf[i] == 0 and buf[i + 1] == 0:
            if buf[i + 2] == 1:
                starts.append((i, 3))
                i += 3
                continue
            if i + 3 < n and buf[i + 2] == 0 and buf[i + 3] == 1:
                starts.append((i, 4))
                i += 4
                continue
        i += 1
    view = memoryview(buf)
    nals: list[memoryview] = []
    for k, (off, prefix_len) in enumerate(starts):
        body_start = off + prefix_len
        body_end = starts[k + 1][0] if k + 1 < len(starts) else n
        if body_end > body_start:
            nals.append(view[body_start:body_end])
    return nals


def _nal_type(nal: memoryview) -> int:
    return nal[0] & 0x1F if nal else 0


class HwH264Encoder:
    """Hardware H.264 encoder backed by ``libsrcampy.Encoder``.

    Not thread-safe. Drive from a single producer thread (the ROS callback
    in this project).
    """

    def __init__(self, channel: int, width: int, height: int) -> None:
        # TODO(mboos): the libsrcampy Encoder Python binding only exposes
        # encode(channel, type, w, h); the underlying VPPEncode C++ class
        # supports h264_cbr_params.bit_rate and gop_params.gop_preset_idx
        # (see /app/multimedia_samples/sample_pipeline/common/vp_codec.c)
        # but they aren't reachable from Python. Until we add a C shim or
        # switch to a Camera+VPS-driven downscale path, the encoder runs at
        # its default bit_rate=8000 kbps with a fixed GOP.
        # Lazy import so test environments that lack the Horizon SDK can still
        # import the project (the wrapper just becomes unusable, not poisoned).
        from hobot_vio import libsrcampy  # noqa: PLC0415

        self._channel = channel
        self._w = width
        self._h = height
        self._aligned_w = (width + 15) & ~15
        self._aligned_h = (height + 15) & ~15
        self._needs_pad = (self._aligned_w, self._aligned_h) != (width, height)
        # The encoder rejects any buffer whose size does not exactly match the
        # aligned dimensions, so we always feed it the padded layout. The Y
        # plane gets copied into rows 0..h and the UV plane into rows
        # aligned_h..aligned_h+h/2; rows beyond are left as zeros.
        if self._needs_pad:
            self._padded = np.zeros(
                (self._aligned_h * 3 // 2, self._aligned_w), dtype=np.uint8,
            )
        else:
            self._padded = None
        self._enc = libsrcampy.Encoder()
        rc = self._enc.encode(channel, _TYPE_H264, self._aligned_w, self._aligned_h)
        if rc != 0:
            raise RuntimeError(
                f"libsrcampy.Encoder.encode init failed: rc={rc} "
                f"channel={channel} {self._aligned_w}x{self._aligned_h}",
            )
        self._sps: bytes | None = None
        self._pps: bytes | None = None
        self._closed = False
        self._last_force_idr = 0.0
        logger.info(
            "HwH264Encoder opened channel=%d %dx%d (aligned from %dx%d, pad=%s)",
            channel, self._aligned_w, self._aligned_h, width, height, self._needs_pad,
        )

    @property
    def width(self) -> int:
        return self._aligned_w

    @property
    def height(self) -> int:
        return self._aligned_h

    @property
    def have_parameter_sets(self) -> bool:
        return self._sps is not None and self._pps is not None

    def force_idr(self) -> None:
        """Reinitialize the VPU encoder so the next frame is an IDR.

        Rate-limited to once per 500 ms to avoid thrashing on burst PLI/FIR.
        """
        now = time.monotonic()
        elapsed = now - self._last_force_idr
        if elapsed < 0.5:
            return
        self._last_force_idr = now
        logger.info("Forcing IDR on VPU encoder channel=%d (last was %.1fs ago)",
                    self._channel, elapsed)
        self._enc.close()
        rc = self._enc.encode(self._channel, _TYPE_H264, self._aligned_w, self._aligned_h)
        if rc != 0:
            raise RuntimeError(
                f"libsrcampy.Encoder re-init failed after force_idr: rc={rc} "
                f"channel={self._channel} {self._aligned_w}x{self._aligned_h}",
            )

    def encode_nv12(self, nv12: np.ndarray) -> list[memoryview]:
        """Push one NV12 frame, return its NAL units (each without start code).

        ``nv12`` must be a numpy array shaped ``(h*3//2, w)`` matching the
        ``width``/``height`` the encoder was opened with.
        """
        if self._needs_pad:
            assert self._padded is not None
            # Y plane: nv12[0:h, 0:w] → padded[0:h, 0:w]
            self._padded[: self._h, : self._w] = nv12[: self._h, : self._w]
            # UV plane: nv12[h:h+h/2, 0:w] → padded[aligned_h:aligned_h+h/2, 0:w]
            uv_h = self._h // 2
            self._padded[self._aligned_h : self._aligned_h + uv_h, : self._w] = (
                nv12[self._h : self._h + uv_h, : self._w]
            )
            buffer = self._padded.tobytes()
        else:
            buffer = nv12.tobytes()
        rc = self._enc.send_frame(buffer)
        if rc != 0:
            logger.warning("send_frame returned non-zero rc=%d", rc)
            return []
        annex_b = self._enc.get_frame()
        if not annex_b:
            return []
        nals = _split_annex_b(annex_b)
        return self._fixup(nals)

    def _fixup(self, nals: list[memoryview]) -> list[memoryview]:
        """Cache SPS/PPS the first time we see them and re-inject before IDRs.

        SPS/PPS are converted to ``bytes`` when cached so they survive after
        the originating ``get_frame()`` buffer is released.  All other NALs
        stay as zero-copy ``memoryview`` slices.
        """
        has_idr = False
        passthrough: list[memoryview] = []
        for nal in nals:
            t = _nal_type(nal)
            if t == _NAL_TYPE_SPS:
                self._sps = bytes(nal)  # must outlive this frame's buffer
                continue
            if t == _NAL_TYPE_PPS:
                self._pps = bytes(nal)
                continue
            if t == _NAL_TYPE_IDR:
                has_idr = True
            passthrough.append(nal)
        if has_idr and self._sps is not None and self._pps is not None:
            return [memoryview(self._sps), memoryview(self._pps), *passthrough]
        return passthrough

    def close(self) -> None:
        if not self._closed:
            try:
                self._enc.close()
            except Exception:  # noqa: BLE001
                logger.exception("libsrcampy.Encoder.close raised")
            self._closed = True

    def __del__(self) -> None:
        self.close()
