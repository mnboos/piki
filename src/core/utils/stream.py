import atexit
import json
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import threading
import time
import traceback
import uuid
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Semaphore
from pathlib import Path
from typing import IO, Any, Optional

import numpy as np
import rclpy
from django.conf import settings as django_settings
from sensor_msgs.msg import Image as RosImage
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from .. import events
from . import shared as _s
from .ai import MODEL_INPUT_TYPE, detect_objects
from .event_payloads import (
    build_mask_payload,
    build_recording_payload,
    build_rois_payload,
    build_splash_payload,
    build_tracker_payload,
)
from .event_log import EventLogger
from .func import (
    slice_roi_into_tiles,
)
from .hw_encoder import HwH264Encoder
from .webrtc import webrtc_publish
from .interfaces import Box, DoubleBuffer
from .metrics import LiveMetricsDashboard
from .recording import (
    EventClipRecorder,
    pre_buffer_append,
    pre_buffer_snapshot,
    write_frame,
)
from .shared import (
    NUM_AI_WORKERS,
    Detection,
    InferenceOutput,
    ai_input_size,
    app_settings,
    bbox_ema_alpha,
    coord_ema_alpha,
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
    min_consecutive_frames,
    motion_detector,
    preview_downscale_factor,
    prob_threshold,
    prob_threshold_keep,
    recording_active,
    servo_aim_confidence,
    replaying_active,
    settings,
    tracker_confirm_hits,
    tracker_delta_t,
    tracker_enabled,
    tracker_inertia,
    tracker_iou_threshold,
    tracker_max_misses,
)

logger = logging.getLogger(__name__)


# Target-lock state: tracks the currently locked detection across YOLO frames so
# the servo doesn't jump when detection order changes or multiple targets exist.
_locked_target_bbox: list[float] | None = None   # normalized [ymin, xmin, ymax, xmax]
_locked_target_label: str | None = None
_locked_target_lost_since: float | None = None   # time.time() when target was last seen
_prev_aim_bbox: list[float] | None = None        # 1-frame delay buffer for servo feed

# --- Phase A stability state (hysteresis, min-streak) ---
_label_streak: dict[str, int] = {}

double_buffer: DoubleBuffer | None = None
worker_semaphore = Semaphore(NUM_AI_WORKERS)
ffmpeg_process: subprocess.Popen | None = None

# Hardware H.264 encoder for WebRTC. Created lazily on the first frame after a
# peer connects so we don't allocate VPU resources when nobody's watching.
# Lives on the ROS callback thread; no lock needed (single producer).
_hw_encoder: HwH264Encoder | None = None
_hw_encoder_dims: tuple[int, int] | None = None

# Monotonic ns of the last encode call — drives the target-FPS frame skip.
_last_encode_ns: int = 0

# When ROS_WEBRTC_TOPIC is set, a dedicated subscription delivers
# hardware-scaled sub-stream frames for WebRTC encoding (Phase 1 of the
# zero-copy pipeline).  process_frame() skips its own encode path so the VPU
# encoder is not shared between the two callbacks.
_WEBRTC_SUBSTREAM_TOPIC: str = os.environ.get("ROS_WEBRTC_TOPIC", "")

# Separate encoder state for the sub-stream path (VPU channel 1).
_hw_encoder_webrtc: HwH264Encoder | None = None
_hw_encoder_webrtc_dims: tuple[int, int] | None = None


def _get_hw_encoder(width: int, height: int) -> HwH264Encoder:
    global _hw_encoder, _hw_encoder_dims
    if _hw_encoder is None or _hw_encoder_dims != (width, height):
        if _hw_encoder is not None:
            _hw_encoder.close()
        _hw_encoder = HwH264Encoder(channel=0, width=width, height=height)
        _hw_encoder_dims = (width, height)
    return _hw_encoder


def _get_hw_encoder_webrtc(width: int, height: int) -> HwH264Encoder:
    """Return the WebRTC-dedicated encoder (VPU channel 1, sub-stream dims)."""
    global _hw_encoder_webrtc, _hw_encoder_webrtc_dims
    if _hw_encoder_webrtc is None or _hw_encoder_webrtc_dims != (width, height):
        if _hw_encoder_webrtc is not None:
            _hw_encoder_webrtc.close()
        _hw_encoder_webrtc = HwH264Encoder(channel=1, width=width, height=height)
        _hw_encoder_webrtc_dims = (width, height)
    return _hw_encoder_webrtc

# Latest foreground motion mask — updated every frame in process_frame() and used
# by on_done() to compute a foreground-weighted aim centroid within the detection
# bbox.  Written and read by the same streaming thread, so no locking is needed.
_latest_mask: "Optional[np.ndarray]" = None
_latest_mask_shape: "tuple[int, int]" = (1, 1)

# Event-triggered recording runtime state.
_event_recorder: "EventClipRecorder | None" = None
_event_recorder_lock = threading.Lock()
_event_clip_until: float = 0.0

_event_logger: "EventLogger | None" = None
_event_frame_idx: int = 0
_event_started_monotonic: float = 0.0
_event_video_path: str = ""

_latest_inference_log_entries: list[dict] = []
_latest_inference_lock = threading.Lock()

# Module-level OC-Sort tracker, lazily constructed on first inference.
# OCSORT is imported lazily (Rust .so) to avoid delaying stream startup.
_tracker: "Optional[Any]" = None
_tracker_params: "Optional[tuple[float, int, int, int, float]]" = None
# Guard concurrent _tracker.update() calls — process_frame() (ROS thread) and
# on_done() (inference thread) both call into the tracker.
_tracker_lock = threading.Lock()
_label_to_id: dict[str, int] = {}
_id_to_label: dict[int, str] = {}

# Per-frame smoothed detection center state for the coord EMA filter.
# Each entry is (cx, cy, label) in normalized coords; matched greedily to the
# next frame's raw detections by label + center distance before tracking.
_smoothed_raw_coords: list[tuple[float, float, str]] = []


def _foreground_centroid(
    bbox_normalized: "list[float]",
    mask: "np.ndarray",
    frame_shape: "tuple[int, ...]",
) -> "tuple[float, float]":
    """Return the foreground-weighted centroid (cx_n, cy_n) within a bbox."""
    ymin, xmin, ymax, xmax = bbox_normalized
    fh, fw = frame_shape[:2]
    y1 = max(0, int(ymin * fh))
    y2 = min(fh, int(ymax * fh))
    x1 = max(0, int(xmin * fw))
    x2 = min(fw, int(xmax * fw))
    if y2 > y1 and x2 > x1:
        roi = mask[y1:y2, x1:x2]
        ys, xs = np.where(roi > 0)
        if xs.size > 0:
            return (float(xs.mean()) + x1) / fw, (float(ys.mean()) + y1) / fh
    return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0


def init_worker():
    pid = os.getpid()
    ppid = os.getppid()
    logger.info(f"[Worker-{pid}]: Setup")

    def f() -> None:
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()

    @atexit.register
    def _cleanup() -> None:
        logger.info(f"[Worker-{pid}] Shutting down....")


class PikiVisionNode(Node):
    def __init__(self):
        super().__init__("piki_vision_node")
        topic_name = os.environ.get("ROS_IMAGE_TOPIC", "/StereoNetNode/rectified_image")

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.get_logger().info(f"Subscribing to: {topic_name}")
        self.subscription = self.create_subscription(RosImage, topic_name, self.listener_callback_hbm, qos_profile)
        self.target_pub = self.create_publisher(String, "/piki/detections", 10)

        if _WEBRTC_SUBSTREAM_TOPIC:
            self.get_logger().info(f"WebRTC sub-stream topic: {_WEBRTC_SUBSTREAM_TOPIC}")
            self._webrtc_subscription = self.create_subscription(
                RosImage, _WEBRTC_SUBSTREAM_TOPIC, self.listener_callback_webrtc, qos_profile,
            )

    def listener_callback_webrtc(self, msg: Any):
        """Encode a hardware-scaled sub-stream frame directly to H.264 for WebRTC.

        This callback is only active when ROS_WEBRTC_TOPIC is set. It runs
        independently of listener_callback_hbm so WebRTC encoding never competes
        with motion detection / inference for CPU time.
        """
        if not _s.webrtc_active.is_set():
            return
        try:
            w, h = msg.width, msg.height
            stride = msg.step if msg.step > 0 else w
            if stride == w:
                data_size = h * w * 3 // 2
                nv12 = np.frombuffer(msg.data, dtype=np.uint8)[:data_size].reshape(h * 3 // 2, w)
            else:
                raw_buffer = (
                    np.frombuffer(msg.data, dtype=np.uint8)
                    if not isinstance(msg.data, np.ndarray)
                    else msg.data
                )
                y_plane = raw_buffer[: h * stride].reshape(h, stride)[:, :w]
                uv_start = h * stride
                uv_plane = raw_buffer[uv_start : uv_start + (h // 2) * stride].reshape(h // 2, stride)[:, :w]
                nv12 = np.vstack([y_plane, uv_plane])
            enc = _get_hw_encoder_webrtc(w, h)
            if _s.webrtc_keyframe_requested.is_set():
                enc.force_idr()
                _s.webrtc_keyframe_requested.clear()
            nals = enc.encode_nv12(nv12)
            if nals:
                webrtc_publish(nals, time.monotonic_ns())
        except Exception:
            logger.exception("WebRTC sub-stream encode failed")

    def listener_callback_hbm(self, msg: Any):
        if replaying_active.is_set():
            return
        try:
            fps_counter.tick()
            events.publish_throttled("tracker_status", build_tracker_payload(), 0.1)
            w, h = msg.width, msg.height
            stride = msg.step if msg.step > 0 else w

            if stride == w:
                data_size = h * w * 3 // 2
                nv12 = np.frombuffer(msg.data, dtype=np.uint8)[:data_size].reshape(h * 3 // 2, w)
            else:
                raw_buffer = np.frombuffer(msg.data, dtype=np.uint8) if not isinstance(msg.data, np.ndarray) else msg.data
                y_plane = raw_buffer[: h * stride].reshape(h, stride)[:, :w]
                uv_start = h * stride
                uv_plane = raw_buffer[uv_start : uv_start + (h // 2) * stride].reshape(h // 2, stride)[:, :w]
                nv12 = np.vstack([y_plane, uv_plane])

            process_frame(nv12_frame=nv12, frame_h=h)

        except Exception as e:
            logger.exception("HBM error")
            self.get_logger().error(f"HBM Fix Error: {e}")
            self.get_logger().error(traceback.format_exc())

    def on_inference_done(self, future: Future, tile_x: int, tile_y: int):
        try:
            if future in active_futures:
                active_futures.remove(future)

            result = future.result()

            mapped_detections = []
            for det in result.detections:
                x1 = det.bbox[0] + tile_x
                y1 = det.bbox[1] + tile_y
                x2 = det.bbox[2] + tile_x
                y2 = det.bbox[3] + tile_y

                bbox_global = (x1, y1, x2 - x1, y2 - y1)

                mapped_detections.append(Detection(label=det.label, confidence=det.confidence, bbox=bbox_global))

        except Exception as e:
            self.get_logger().error(f"Inference Result Error: {e}")

    def publish_detections(self, payload: list):
        """Publish detection dicts (label/confidence/bbox normalised [xmin,ymin,xmax,ymax]) on /piki/detections."""
        if not self.context.ok():
            return
        msg = String()
        msg.data = json.dumps(payload)
        self.target_pub.publish(msg)


inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piki-inference")
thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piki-streamer")
ros_node: Optional["PikiVisionNode"] = None


def frame_producer(ffmpeg_stdout: IO[bytes], buffer_instance: DoubleBuffer):
    shape = buffer_instance.shape
    dtype = buffer_instance.dtype
    frame_size = int(np.prod(shape) * np.dtype(dtype).itemsize)
    logger.info("Producer started. Waiting for data from FFmpeg stdout...")
    try:
        while True:
            in_bytes = ffmpeg_stdout.read(frame_size)
            if not in_bytes or len(in_bytes) != frame_size:
                logger.warning("Producer received incomplete frame from FFmpeg. Shutting down.")
                break
            frame = np.frombuffer(in_bytes, dtype=dtype).reshape(shape)
            buffer_instance.write(frame)
    except:
        logger.exception("Exception in producer thread:")
        traceback.print_exc()
        raise
    logger.info("Producer finished.")


active_futures: list[Future[InferenceOutput]] = []
_ai_tiles_saved = 0
max_output_timestamp = 0
dashboard = LiveMetricsDashboard()


def get_measure(description: str):
    now = time.perf_counter()

    def measure(*, log: bool = True) -> None:
        end = time.perf_counter()
        ms = str(round((end - now) * 1000, 2)).rjust(5)
        if log:
            logger.info(f"{description}: {ms} ms!")

    return measure


def _compute_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter_ymin = max(a[0], b[0])
    inter_xmin = max(a[1], b[1])
    inter_ymax = min(a[2], b[2])
    inter_xmax = min(a[3], b[3])
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter = inter_h * inter_w
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_object_detection(
    frame_hires: np.ndarray,
    rois: list[Box],
    timestamp: int,
) -> InferenceOutput:
    worker_pid = mp.current_process().pid or 0
    if is_object_detection_disabled.is_set():
        return InferenceOutput(worker_pid=worker_pid, timestamp=timestamp, avg_duration=0, detections=[])

    _profile = bool(os.environ.get("PIKI_PROFILE"))

    try:
        frame_w = frame_hires.shape[1]
        frame_h = frame_hires.shape[0] * 2 // 3


        _t = time.perf_counter()
        tiles = slice_roi_into_tiles(
            frame=frame_hires,
            rois=rois,
            tile_size=ai_input_size,
            preview_downscale_factor=preview_downscale_factor,
            model_input_type=MODEL_INPUT_TYPE,
        )
        if _profile:
            logger.info("PERF stage=tile_slice ms=%.2f tiles=%d", (time.perf_counter() - _t) * 1000, len(tiles))

        total_duration = 0
        all_detections: list[Detection] = []

        for tile_img, tile_x, tile_y in tiles:
            # Save first 3 tiles for visual inspection.
            global _ai_tiles_saved
            if _ai_tiles_saved < 3:
                try:
                    nv12_2d = tile_img.reshape(ai_input_size * 3 // 2, ai_input_size)
                    bgr = cv2.cvtColor(nv12_2d, cv2.COLOR_YUV2BGR_NV12)
                    out = Path(django_settings.MEDIA_ROOT) / f"ai_tile_{_ai_tiles_saved + 1}.jpg"
                    ok = cv2.imwrite(str(out), bgr)
                    logger.info("Saved AI tile %d to %s (ok=%s, shape=%s, size=%d)",
                                _ai_tiles_saved + 1, out, ok, nv12_2d.shape, tile_img.size)
                    _ai_tiles_saved += 1
                except Exception as exc:
                    logger.warning("Failed to save AI tile %d: %s (tile_img size=%d)",
                                   _ai_tiles_saved + 1, exc, tile_img.size)

            duration, detections = detect_objects(tile_img)
            total_duration += duration

            for label, confidence, local_pixel_bbox in detections:
                local_px_xmin, local_px_ymin, local_px_xmax, local_px_ymax = local_pixel_bbox

                global_px_xmin = local_px_xmin + tile_x
                global_px_ymin = local_px_ymin + tile_y
                global_px_xmax = local_px_xmax + tile_x
                global_px_ymax = local_px_ymax + tile_y

                final_norm_coords = [
                    global_px_ymin / frame_h,
                    global_px_xmin / frame_w,
                    global_px_ymax / frame_h,
                    global_px_xmax / frame_w,
                ]
                all_detections.append(Detection(label=label, confidence=confidence, bbox=final_norm_coords))

        avg_duration = 0 if not tiles else total_duration // len(tiles)
        return InferenceOutput(
            worker_pid=worker_pid,
            timestamp=timestamp,
            avg_duration=avg_duration,
            detections=all_detections,
        )

    except:
        logger.exception("Fatal error in inference thread (PID: %d)", os.getpid())
        raise


def on_done(future: Future[InferenceOutput]):
    """Handle completed inference."""
    global max_output_timestamp, _locked_target_bbox, _locked_target_label, _locked_target_lost_since, _prev_aim_bbox, _smoothed_raw_coords
    active_futures.remove(future)
    try:
        worker_pid, timestamp, inference_time, detections = future.result()
        logger.info("YOLO_DONE ts=%d raw_detections=%d duration_ms=%d", timestamp, len(detections), inference_time)

        if timestamp < max_output_timestamp:
            logger.info(f"Inference result arrived late, discarding (timestamp={timestamp})")
            return

        max_output_timestamp = timestamp

        # Exclusion-zone filter
        from . import exclusion as _exclusion  # noqa: PLC0415

        zones_active = _exclusion.has_zones()
        log_entries: list[dict] = []
        kept: list = []
        n_dropped_by_zone = 0
        for label, confidence, bbox_normalized in detections:
            inside = (
                _exclusion.bbox_centroid_inside_any(bbox_normalized)
                if zones_active else False
            )
            log_entries.append({
                "label": label,
                "confidence": float(confidence),
                "bbox_norm": [float(v) for v in bbox_normalized],
                "inside_exclusion": bool(inside),
                "track_id": None,
                "track_age_frames": None,
            })
            if not inside:
                kept.append((label, confidence, bbox_normalized))
            else:
                n_dropped_by_zone += 1
        detections = kept
        if n_dropped_by_zone:
            logger.info(
                "Exclusion filter dropped %d/%d detection(s) inside zones.",
                n_dropped_by_zone, n_dropped_by_zone + len(kept),
            )

        # --- Coord EMA smoothing (applied before tracker) ---
        # Smooths the mask centroid (x, y) of each raw detection using:
        #   smoothed = alpha * new + (1 - alpha) * old
        # Detections are matched to the previous frame's state by label +
        # nearest centroid distance (greedy).  Works with tracker on or off.
        # _latest_mask is always set when on_done() fires (inference only runs
        # after movement is detected); _foreground_centroid falls back to the
        # bbox midpoint internally if the bbox region has no foreground pixels.
        c_alpha = max(0.0, min(1.0, float(coord_ema_alpha.value)))
        smoothed_centers: list[tuple[float, float]] = []
        if c_alpha < 0.999 and kept:
            new_state: list[tuple[float, float, str]] = []
            smoothed_kept = []
            used_prev: set[int] = set()
            for label, confidence, bbox_norm in kept:
                ymin, xmin, ymax, xmax = bbox_norm
                cx_raw, cy_raw = _foreground_centroid(bbox_norm, _latest_mask, _latest_mask_shape)
                w = xmax - xmin
                h = ymax - ymin
                threshold = (w + h) / 2.0
                best_i, best_dist = None, float("inf")
                for i, (pcx, pcy, plabel) in enumerate(_smoothed_raw_coords):
                    if i in used_prev or plabel != label:
                        continue
                    dist = ((cx_raw - pcx) ** 2 + (cy_raw - pcy) ** 2) ** 0.5
                    if dist < threshold and dist < best_dist:
                        best_dist, best_i = dist, i
                if best_i is not None:
                    used_prev.add(best_i)
                    cx = c_alpha * cx_raw + (1.0 - c_alpha) * _smoothed_raw_coords[best_i][0]
                    cy = c_alpha * cy_raw + (1.0 - c_alpha) * _smoothed_raw_coords[best_i][1]
                else:
                    cx, cy = cx_raw, cy_raw
                new_state.append((cx, cy, label))
                smoothed_centers.append((cx, cy))
                smoothed_kept.append((label, confidence, [cy - h / 2.0, cx - w / 2.0, cy + h / 2.0, cx + w / 2.0]))
            _smoothed_raw_coords = new_state
            kept = smoothed_kept
        else:
            new_state = []
            for label, confidence, bbox_norm in kept:
                cx, cy = _foreground_centroid(bbox_norm, _latest_mask, _latest_mask_shape)
                new_state.append((cx, cy, label))
                smoothed_centers.append((cx, cy))
            _smoothed_raw_coords = new_state
        # Propagate smoothed coords back into log_entries so det_payload and
        # event-log entries reflect the smoothed positions.
        ki = 0
        for ent in log_entries:
            if not ent["inside_exclusion"]:
                ent["bbox_norm"] = [float(v) for v in kept[ki][2]]
                ent["center_norm"] = [float(smoothed_centers[ki][0]), float(smoothed_centers[ki][1])]
                ki += 1
        detections = kept

        # --- OC-Sort tracker (Phase B) ---
        global _tracker, _tracker_params, _label_to_id, _id_to_label
        tracker_on = bool(tracker_enabled.value)
        if tracker_on:
            desired_params = (
                float(tracker_iou_threshold.value),
                int(tracker_max_misses.value),
                max(1, int(tracker_confirm_hits.value)),
                int(tracker_delta_t.value),
                float(tracker_inertia.value),
            )
            with _tracker_lock:
                if _tracker is None or _tracker_params != desired_params:
                    from trackforge import OCSORT  # noqa: PLC0415 — lazy load Rust .so

                    _tracker = OCSORT(
                        max_age=desired_params[1],
                        min_hits=desired_params[2],
                        iou_threshold=desired_params[0],
                        delta_t=desired_params[3],
                        inertia=desired_params[4],
                    )
                    _tracker_params = desired_params

                # Build detections in OCSORT format: ([x, y, w, h], score, class_id)
                tf_dets: list[tuple[list[float], float, int]] = []
                for label, conf, (ymin, xmin, ymax, xmax) in detections:
                    cls_id = _label_to_id.setdefault(label, len(_label_to_id))
                    _id_to_label[cls_id] = label
                    tlwh = [xmin, ymin, xmax - xmin, ymax - ymin]
                    tf_dets.append((tlwh, float(conf), cls_id))
                tracked = _tracker.update(tf_dets)

            visible: list[tuple[str, float, list[float], int, int]] = []
            for track_id, tlwh, score, cls_id in tracked:
                x, y, w, h = tlwh
                bbox_norm = [y, x, y + h, x + w]
                if zones_active and _exclusion.bbox_centroid_inside_any(bbox_norm):
                    continue
                label = _id_to_label.get(cls_id, "unknown")
                visible.append((label, float(score), bbox_norm, int(track_id), 0))

            for ent in log_entries:
                if ent["inside_exclusion"]:
                    continue
                best_iou = float(tracker_iou_threshold.value)
                best: Optional[tuple[int, int]] = None
                lbl_lc = ent["label"].strip().lower()
                for label, _conf, bbox_norm, tid, age in visible:
                    if label.strip().lower() != lbl_lc:
                        continue
                    iou_v = _compute_iou(ent["bbox_norm"], bbox_norm)
                    if iou_v > best_iou:
                        best_iou = iou_v
                        best = (tid, age)
                if best is not None:
                    ent["track_id"] = int(best[0])
                    ent["track_age_frames"] = int(best[1])

            detections = [(label, conf, bbox_norm, tid, age) for label, conf, bbox_norm, tid, age in visible]

        # Push detections to any connected SPA clients. We publish every raw
        # AI hit that survived the exclusion filter (with the matched track id
        # attached when the tracker has one) — that way unconfirmed objects
        # still render as boxes immediately. The tracker output (`detections`
        # below) continues to drive servo aiming and event recording.
        # Normalized bbox is converted from internal [ymin, xmin, ymax, xmax]
        # to wire-friendly [xmin, ymin, xmax, ymax].
        det_payload: list[dict] = []
        for ent in log_entries:
            if ent["inside_exclusion"]:
                continue
            ymin, xmin, ymax, xmax = ent["bbox_norm"]
            tid = ent.get("track_id")
            cx, cy = ent["center_norm"]
            det_payload.append({
                "tid": int(tid) if tid is not None else None,
                "label": str(ent["label"]),
                "score": float(ent["confidence"]),
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "center": [float(cx), float(cy)],
            })
        events.publish("detections", {
            "frame_ts_ns": int(timestamp),
            "detections": det_payload,
        })
        if log_entries:
            logger.info(
                "YOLO_PUBLISH ts=%d log_entries=%d published=%d in_exclusion=%d tracker=%s",
                timestamp,
                len(log_entries),
                len(det_payload),
                sum(1 for e in log_entries if e["inside_exclusion"]),
                "on" if tracker_on else "off",
            )

        with _latest_inference_lock:
            global _latest_inference_log_entries
            _latest_inference_log_entries = log_entries

        aim_enabled = app_settings.aim_settings.servo_enabled
        target_classes = {c.strip().lower() for c in (app_settings.aim_settings.target_classes or [])}
        target_lock_duration = float(app_settings.aim_settings.target_lock_duration)

        conf_enter = float(prob_threshold.value)
        conf_keep = float(prob_threshold_keep.value)
        if conf_keep > conf_enter:
            conf_keep = conf_enter

        # Servo-aiming confidence thresholds — independent of display/recording.
        aim_conf = float(servo_aim_confidence.value)
        aim_conf_keep = max(conf_keep, aim_conf * 0.6)
        min_streak_required = max(1, int(min_consecutive_frames.value))
        ema_alpha = max(0.0, min(1.0, float(bbox_ema_alpha.value)))

        firing_labels: set[str] = {
            label.strip().lower()
            for label, confidence, _bbox, *_ in detections
            if confidence >= conf_enter
        }
        for lbl in list(_label_streak.keys()):
            if lbl in firing_labels:
                _label_streak[lbl] = min(_label_streak[lbl] + 1, min_streak_required + 5)
            else:
                new_val = _label_streak[lbl] - 1
                if new_val <= 0:
                    del _label_streak[lbl]
                else:
                    _label_streak[lbl] = new_val
        for lbl in firing_labels:
            _label_streak.setdefault(lbl, 1)

        def _confirmed(label: str) -> bool:
            return _label_streak.get(label.strip().lower(), 0) >= min_streak_required

        # --- Servo aiming with hysteresis-aware target lock ---
        if (_locked_target_bbox is not None
                and zones_active
                and _exclusion.bbox_centroid_inside_any(_locked_target_bbox)):
            logger.info("Locked target entered exclusion zone — releasing lock and homing servo.")
            _locked_target_bbox = None
            _locked_target_label = None
            _locked_target_lost_since = None
            _prev_aim_bbox = None
            if aim_enabled:
                _release_servo_lock()

        matching: list[tuple[str, float, list[float]]] = [
            (label, confidence, bbox_normalized)
            for label, confidence, bbox_normalized, *_ in detections
            if not target_classes or label.strip().lower() in target_classes
        ]

        chosen: tuple[str, float, list[float]] | None = None

        if _locked_target_bbox is not None:
            best_iou = 0.0
            best: tuple[str, float, list[float]] | None = None
            for det in matching:
                if det[1] < aim_conf_keep:
                    continue
                iou = _compute_iou(_locked_target_bbox, det[2])
                if iou > best_iou:
                    best_iou = iou
                    best = det
            if best is not None and best_iou >= 0.3:
                chosen = best
                _locked_target_lost_since = None
            else:
                if _locked_target_lost_since is None:
                    _locked_target_lost_since = time.time()
                elapsed = time.time() - _locked_target_lost_since
                if elapsed >= target_lock_duration:
                    logger.info("Target lock expired after %.1fs.", elapsed)
                    _locked_target_bbox = None
                    _locked_target_label = None
                    _locked_target_lost_since = None
                    _prev_aim_bbox = None
                    if aim_enabled:
                        _release_servo_lock()

        if chosen is None and _locked_target_bbox is None:
            for det in matching:
                label, conf, _bbox = det
                if conf >= aim_conf and _confirmed(label):
                    chosen = det
                    break

        if chosen is not None:
            chosen_label, chosen_confidence, chosen_bbox = chosen
            new_bbox = list(chosen_bbox)
            if _locked_target_bbox is None or ema_alpha >= 0.999:
                _locked_target_bbox = new_bbox
            else:
                _locked_target_bbox = [
                    (1.0 - ema_alpha) * old + ema_alpha * meas
                    for old, meas in zip(_locked_target_bbox, new_bbox)
                ]
            _locked_target_label = chosen_label

            smoothed_bbox = list(_locked_target_bbox)

            # Feed the servo loop with a 1-frame delay so single-frame
            # detection outliers never reach the Kalman.  At 30 fps that
            # is 33 ms — imperceptible to a human observer.
            if aim_enabled:
                from .engine import feed_target  # noqa: PLC0415

                aim_bbox = _prev_aim_bbox if _prev_aim_bbox is not None else smoothed_bbox
                _aim_center = None
                if _latest_mask is not None:
                    _aim_center = _foreground_centroid(aim_bbox, _latest_mask, _latest_mask_shape)
                feed_target(bbox_normalized=aim_bbox, aim_center=_aim_center)
            _prev_aim_bbox = smoothed_bbox

            # --- Splash logic: fire relay when a splash-class target is locked ---

            if _s.splash_enabled.is_set() and time.time() > _s.splash_cooldown_until:
                splash_classes = target_classes
                if not splash_classes or chosen_label.strip().lower() in splash_classes:
                    if _s.splash_armed_at == 0.0:
                        _s.splash_armed_at = time.time()
                        logger.info("Splash armed for target=%s (delay=%.1fs)", chosen_label, _s.splash_delay.value)
                        events.publish("splash_status", build_splash_payload())
                    elif time.time() - _s.splash_armed_at >= _s.splash_delay.value:
                        from .engine import activate_pump  # noqa: PLC0415
                        _s.splash_cooldown_until = time.time() + _s.splash_cooldown.value
                        _s.splash_firing_until = time.time() + _s.splash_duration.value
                        _s.splash_armed_at = 0.0
                        activate_pump(_s.splash_duration.value, _s.pump_duty.value)
                        logger.info("Splash fired for target=%s (duration=%.1fs, cooldown=%.1fs)",
                                    chosen_label, _s.splash_duration.value, _s.splash_cooldown.value)
                        events.publish("splash_status", build_splash_payload())
                        if _event_logger is not None:
                            _event_logger.write({
                                "event": "splash_fired",
                                "label": chosen_label,
                                "duration_seconds": float(_s.splash_duration.value),
                            })
                else:
                    if _s.splash_armed_at > 0:
                        _s.splash_armed_at = 0.0
                        events.publish("splash_status", build_splash_payload())
            else:
                if _s.splash_armed_at > 0:
                    _s.splash_armed_at = 0.0
                    events.publish("splash_status", build_splash_payload())

        dashboard.update(worker_id=worker_pid, inference_time=inference_time)

        # Mirror the normalised detection payload to the /piki/detections ROS
        # topic for external consumers. Frontend already received it via WS.
        if ros_node is not None:
            ros_node.publish_detections(det_payload)

        # --- Event-triggered recording ---
        if (event_recording_enabled.is_set()
                and not event_recording_active.is_set()
                and not recording_active.is_set()
                and time.time() > event_recording_cooldown_until):
            with event_trigger_classes_lock:
                trigger_set = {c.strip().lower() for c in event_trigger_classes}
            if trigger_set:
                for label, confidence, bbox_normalized, *_ in detections:
                    label_lc = label.strip().lower()
                    if (confidence >= conf_enter
                            and label_lc in trigger_set
                            and _confirmed(label_lc)):
                        _start_event_recording()
                        break

    except KeyboardInterrupt:
        logger.info("Shutting down on KeyboardInterrupt in on_done.")
    except:
        traceback.print_exc()
        raise


def _release_servo_lock() -> None:
    """Tell the servo loop the lock is gone — it will home the servos."""
    try:
        from .engine import release_target  # noqa: PLC0415
        release_target()
    except Exception:
        logger.exception("Failed to release servo lock.")


def _log_live_event_frame(mask) -> None:
    global _event_frame_idx
    if _event_logger is None:
        return
    with _latest_inference_lock:
        detections_payload = list(_latest_inference_log_entries)
    motion_px = int(cv2.countNonZero(mask)) if mask is not None else 0
    locked_label = _locked_target_label
    if _locked_target_lost_since is not None:
        lost_ms = int((time.time() - _locked_target_lost_since) * 1000)
    else:
        lost_ms = None
    _event_logger.write({
        "event": "frame",
        "frame_idx": _event_frame_idx,
        "prebuffer": False,
        "motion": {"pixel_count": motion_px},
        "detections": detections_payload,
        "servo": {
            "pan_deg": float(_s.servo_pan.value),
            "tilt_deg": float(_s.servo_tilt.value),
            "locked_label": locked_label,
            "locked_lost_since_ms": lost_ms,
        },
    })
    _event_frame_idx += 1


def _start_event_recording() -> None:
    global _event_recorder, _event_clip_until
    global _event_logger, _event_frame_idx, _event_started_monotonic, _event_video_path



    pre_frames = pre_buffer_snapshot()

    if pre_frames:
        h, w = pre_frames[0].shape[:2]
    else:
        h, w = 360, 640

    videos_dir = Path(django_settings.MEDIA_ROOT) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    event_id = f"{ts}_{uuid.uuid4().hex[:8]}"
    video_path = str(videos_dir / f"event_{event_id}.mp4")
    log_path = str(videos_dir / f"event_{event_id}.log.jsonl")

    actual_fps = fps_counter.fps if fps_counter.fps > 0 else 30.0


    with event_trigger_classes_lock:
        trigger_classes_snapshot = list(event_trigger_classes)
    trigger_detection = None
    with _latest_inference_lock:
        for ent in _latest_inference_log_entries:
            if ent["inside_exclusion"]:
                continue
            if ent["confidence"] >= float(prob_threshold.value) and (
                not trigger_classes_snapshot
                or ent["label"].strip().lower() in {c.strip().lower() for c in trigger_classes_snapshot}
            ):
                trigger_detection = {
                    "label": ent["label"],
                    "confidence": ent["confidence"],
                    "bbox_norm": ent["bbox_norm"],
                }
                break

    config_snapshot = {
        "conf_threshold_enter": float(prob_threshold.value),
        "conf_threshold_keep": float(prob_threshold_keep.value),
        "min_consecutive_frames": int(min_consecutive_frames.value),
        "bbox_ema_alpha": float(bbox_ema_alpha.value),
        "pre_buffer_seconds": float(event_pre_buffer_seconds.value),
        "post_trigger_seconds": float(event_post_trigger_seconds.value),
        "cooldown_seconds": float(event_cooldown_seconds.value),
        "trigger_classes": trigger_classes_snapshot,
    }

    with _event_recorder_lock:
        _event_recorder = EventClipRecorder(video_path, fps=actual_fps, pre_frames=pre_frames,
                                             frame_w=w, frame_h=h)
        _event_clip_until = time.monotonic() + event_post_trigger_seconds.value
        _event_started_monotonic = time.monotonic()
        _event_video_path = _event_recorder._output_path
        _event_frame_idx = 0
        try:
            _event_logger = EventLogger(log_path, event_id=event_id)
        except Exception:
            logger.exception("Failed to open EventLogger at %s — recording continues without log.", log_path)
            _event_logger = None
        event_recording_active.set()
        events.publish("recording_status", build_recording_payload())

    if _event_logger is not None:
        _event_logger.write({
            "event": "event_started",
            "event_id": event_id,
            "video_filename": Path(_event_video_path).name,
            "fps_estimate": float(actual_fps),
            "frame": {"w": int(w), "h": int(h)},
            "trigger": trigger_detection,
            "config_snapshot": config_snapshot,
            "servo": {"pan_deg": float(_s.servo_pan.value), "tilt_deg": float(_s.servo_tilt.value)},
        })

        for _i in range(len(pre_frames)):
            _event_logger.write({
                "event": "frame",
                "frame_idx": _event_frame_idx,
                "prebuffer": True,
                "motion": None,
                "detections": [],
                "servo": {"pan_deg": float(_s.servo_pan.value), "tilt_deg": float(_s.servo_tilt.value)},
            })
            _event_frame_idx += 1

    logger.info("Event recording started: %s (%d pre-buffer frames, %ds post-trigger, log=%s)",
                _event_video_path, len(pre_frames), event_post_trigger_seconds.value,
                "yes" if _event_logger else "no")


def _finalize_event_recording(*, reason: str = "post_trigger_elapsed") -> None:
    global _event_recorder, _event_clip_until, _event_logger, _event_frame_idx


    with _event_recorder_lock:
        if _event_recorder is None:
            event_recording_active.clear()
            events.publish("recording_status", build_recording_payload())
            return
        final_path, frame_count, error = _event_recorder.close()
        _event_recorder = None
        ev_logger = _event_logger
        _event_logger = None
        duration_seconds = time.monotonic() - _event_started_monotonic if _event_started_monotonic else 0.0
        frames_written = _event_frame_idx

    _event_clip_until = 0.0
    event_recording_active.clear()
    _s.event_recording_cooldown_until = time.time() + event_cooldown_seconds.value
    events.publish("recording_status", build_recording_payload())

    # Capture log_path and event_id BEFORE close() so post-close attribute
    # access doesn't depend on EventLogger's close() implementation.
    log_path = Path(ev_logger.path) if ev_logger is not None else None
    event_id_str = ev_logger.event_id if ev_logger is not None else None

    if ev_logger is not None:
        ev_logger.write({
            "event": "event_ended",
            "reason": reason,
            "frames_written": int(frames_written),
            "duration_seconds": round(float(duration_seconds), 3),
            "error": error,
        })
        ev_logger.close()

    if final_path and frame_count > 0:
        file_path = Path(final_path)
        try:
            from ..models import Video  # noqa: PLC0415

            video_kwargs = {
                "filename": file_path.name,
                "file": str(file_path.relative_to(django_settings.MEDIA_ROOT)),
                "size_bytes": file_path.stat().st_size,
                "source": "event",
            }
            if event_id_str is not None:
                video_kwargs["event_id"] = event_id_str
                if log_path is not None and log_path.exists():
                    video_kwargs["log_file"] = str(log_path.relative_to(django_settings.MEDIA_ROOT))
            video = Video.objects.create(**video_kwargs)
            new_clip = {
                "filename": file_path.name,
                "file": str(file_path.relative_to(django_settings.MEDIA_ROOT)),
                "frame_count": frame_count,
                "time": datetime.now().isoformat(),
                "video_id": video.id,
            }
            with event_clip_queue_lock:
                event_clip_queue.append(new_clip)
            events.publish("event_clips", {"clip": new_clip})
            logger.info("Event clip saved: %s (%d frames, log=%s)", file_path.name, frame_count,
                        log_path.name if log_path else "no")
        except Exception:
            logger.exception("Failed to save event clip to DB")
    else:
        logger.warning("Event recording produced no output: error=%s", error)


def _submit_yolo(*, nv12_frame: np.ndarray, rois: list, timestamp: int) -> None:
    logger.info("YOLO_SUBMIT rois=%d ts=%d", len(rois), timestamp)
    future = inference_pool.submit(run_object_detection, frame_hires=nv12_frame, rois=rois, timestamp=timestamp)
    active_futures.append(future)
    future.add_done_callback(on_done)


# Minimum contour area (in lores pixels) before we send it as a mask polygon.
# Filters specks below the noise floor; matches the magnitude of `min_area`.
_MASK_MIN_CONTOUR_AREA = 20


def _publish_motion_overlays(*, mask: "Optional[np.ndarray]", rois: list, frame_lores: np.ndarray) -> None:
    """Publish ROIs + mask polygons over WebSocket for the SPA overlay.

    Both topics throttle independently (10 Hz / 5 Hz) — the canvas just renders
    the latest payload, so dropping intermediate frames is fine.
    """
    fh, fw = frame_lores.shape[:2]
    inv_w = 1.0 / fw if fw else 1.0
    inv_h = 1.0 / fh if fh else 1.0

    rois_norm = [
        [rx * inv_w, ry * inv_h, rw * inv_w, rh * inv_h]
        for rx, ry, rw, rh in rois
    ]
    events.publish_throttled("rois", build_rois_payload(rois_norm), 0.1)

    polygons: list[list[float]] = []
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        for c in contours:
            if cv2.contourArea(c) < _MASK_MIN_CONTOUR_AREA:
                continue
            pts = c.reshape(-1, 2)
            flat: list[float] = []
            for x, y in pts:
                flat.append(float(x) * inv_w)
                flat.append(float(y) * inv_h)
            polygons.append(flat)
    events.publish_throttled("mask", build_mask_payload(polygons), 0.2)


def process_frame(*, nv12_frame: np.ndarray, frame_h: int):
    global _latest_mask, _latest_mask_shape

    current_time = time.monotonic_ns()

    # Hardware H.264 encode for any connected WebRTC peers. Runs on the VPU,
    # so this should be a fast call that doesn't impact motion/inference timing.
    # Frame skipping honours `webrtc_target_fps` (set to camera rate to disable).
    # Skip this entire block when ROS_WEBRTC_TOPIC is set — in that case
    # listener_callback_webrtc encodes the dedicated sub-stream independently,
    # freeing process_frame from any VPU contention.
    if _s.webrtc_active.is_set() and not _WEBRTC_SUBSTREAM_TOPIC:
        global _last_encode_ns
        target_fps = max(1, int(_s.webrtc_target_fps.value))
        min_interval_ns = 1_000_000_000 // target_fps
        # Subtract a small slack so the actual delivered rate matches the
        # target instead of consistently undershooting by one frame interval.
        if current_time - _last_encode_ns >= min_interval_ns - 5_000_000:
            try:
                frame_w = nv12_frame.shape[1]
                enc = _get_hw_encoder(frame_w, frame_h)
                if _s.webrtc_keyframe_requested.is_set():
                    enc.force_idr()
                    _s.webrtc_keyframe_requested.clear()
                nals = enc.encode_nv12(nv12_frame)
                if nals:
                    webrtc_publish(nals, current_time)
                    _last_encode_ns = current_time
            except Exception:
                logger.exception("WebRTC hardware-encode failed")

    y_plane = nv12_frame[:frame_h]
    step = preview_downscale_factor
    frame_lores = y_plane[::step, ::step]
    has_movement, mask = motion_detector.is_moving(frame_lores)

    if mask is not None and has_movement:
        from . import exclusion as _exclusion  # noqa: PLC0415

        allowed = _exclusion.allowed_mask_for(mask.shape[:2])
        if allowed is not None:
            cv2.bitwise_and(mask, allowed, dst=mask)
            has_movement = bool(cv2.countNonZero(mask) >= settings.foreground_mask_options.pixelcount_threshold.value)

    _latest_mask = mask
    _latest_mask_shape = mask.shape[:2] if mask is not None else (1, 1)

    frame_bgr = cv2.cvtColor(frame_lores, cv2.COLOR_GRAY2BGR)

    if event_recording_enabled.is_set():
        pre_buffer_append(frame_bgr, event_pre_buffer_seconds.value)

    if recording_active.is_set():
        write_frame(frame_bgr)

    if event_recording_active.is_set():
        wrote_frame = False
        with _event_recorder_lock:
            if _event_recorder is not None:
                _event_recorder.write_frame(frame_bgr)
                wrote_frame = True
        if wrote_frame:
            _log_live_event_frame(mask)
        if time.monotonic() >= _event_clip_until:
            _finalize_event_recording()

    rois: list = []
    if has_movement:
        rois = motion_detector.create_rois(mask=mask)
        if rois and not active_futures and not os.environ.get("DISABLE_AI"):
            try:
                _submit_yolo(nv12_frame=nv12_frame, rois=rois, timestamp=time.monotonic_ns())
            except Exception:
                logger.exception("Error in process_frame AI logic")

    _publish_motion_overlays(mask=mask if has_movement else None, rois=rois, frame_lores=frame_lores)


def stream_nonblocking():
    thread_pool.submit(stream_with_ros)


def stream_with_ros():
    try:
        global ros_node

        logger.info("Starting ROS 2 videostream...")

        print("------------!!!!!!!!!!!!! STREAM (ROS 2)")

        rclpy.init(args=None)
        ros_node = PikiVisionNode()

        # Start the decoupled servo command loop.
        from .engine import start_servo_loop  # noqa: PLC0415
        start_servo_loop()

        rclpy.spin(ros_node)
    except Exception:
        logger.exception("ROS 2 streaming failed")
        traceback.print_exc()
        raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()


@atexit.register
def cleanup():
    logger.info("[DJANGO SHUTDOWN] Stopping processes.....")

    try:
        from .engine import stop_servo_loop  # noqa: PLC0415
        stop_servo_loop()
    except Exception:
        logger.exception("Failed to stop servo loop on shutdown.")

    thread_pool.shutdown(wait=True, cancel_futures=True)
    inference_pool.shutdown(wait=True, cancel_futures=True)
    if ffmpeg_process:
        ffmpeg_process.kill()
    if _hw_encoder is not None:
        _hw_encoder.close()
    if _hw_encoder_webrtc is not None:
        _hw_encoder_webrtc.close()

    logger.info("[DJANGO SHUTDOWN] Processes stopped..")
