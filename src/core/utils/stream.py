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
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import Lock, Semaphore
from typing import IO, Any, Optional

import numpy as np
import rclpy
from sensor_msgs.msg import Image as RosImage
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .func import (
    slice_roi_into_tiles,
)
from .interfaces import Box, DoubleBuffer
from .metrics import LiveMetricsDashboard
from .shared import (
    NUM_AI_WORKERS,
    Detection,
    InferenceOutput,
    ai_input_size,
    app_settings,
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
    latest_debug_frame,
    latest_frame,
    mask_transparency,
    motion_detector,
    preview_downscale_factor,
    prob_threshold,
    recording_active,
    replaying_active,
    streaming_active,
)

logger = logging.getLogger(__name__)


# Target-lock state: tracks the currently locked detection across YOLO frames so
# the servo doesn't jump when detection order changes or multiple targets exist.
_locked_target_bbox: list[float] | None = None   # normalized [ymin, xmin, ymax, xmax]
_locked_target_label: str | None = None
_locked_target_lost_since: float | None = None   # time.time() when target was last seen

double_buffer: DoubleBuffer | None = None
worker_semaphore = Semaphore(NUM_AI_WORKERS)
ffmpeg_process: subprocess.Popen | None = None
lowres_frame_cache = {}  # timestamp → lores frame shape (tuple); used by on_done to denormalize bboxes
cache_lock = Lock()

latest_ai_detections = []
latest_ai_lock = threading.Lock()

# Latest foreground motion mask — updated every frame in process_frame() and used
# by on_done() to compute a foreground-weighted aim centroid within the detection
# bbox.  Written and read by the same streaming thread, so no locking is needed.
_latest_mask: "Optional[np.ndarray]" = None
_latest_mask_shape: "tuple[int, int]" = (1, 1)

# Event-triggered recording runtime state.
_event_recorder: "EventClipRecorder | None" = None
_event_recorder_lock = threading.Lock()
_event_clip_until: float = 0.0


def _foreground_centroid(
    bbox_normalized: "list[float]",
    mask: "np.ndarray",
    frame_shape: "tuple[int, ...]",
) -> "tuple[float, float]":
    """Return the foreground-weighted centroid (cx_n, cy_n) within a bbox.

    Crops the binary motion mask to the bbox region and returns the mean x/y
    of all foreground pixels, normalised to ``[0, 1]``.  Falls back to the
    geometric bbox centre when no foreground pixels exist inside the region
    (e.g. the target is stationary, or the mask is stale).
    """
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
        debug_topic = os.environ.get("ROS_DEBUG_TOPIC", "/image_right_raw")

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.get_logger().info(f"Subscribing to: {topic_name}")
        self.subscription = self.create_subscription(RosImage, topic_name, self.listener_callback_hbm, qos_profile)
        self.target_pub = self.create_publisher(String, "/piki/detections", 10)

        # Subscribe to a separate debug/raw topic for the debug video feed.
        # Defaults to the stereonet rectified image so you can compare raw vs rectified.
        self.get_logger().info(f"Debug feed subscribing to: {debug_topic}")
        self.create_subscription(RosImage, debug_topic, self._debug_frame_callback, qos_profile)

    def _debug_frame_callback(self, msg: Any):
        """Receive the debug/raw topic and push its Y-plane into latest_debug_frame."""
        try:
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
            y = nv12[:h]  # luma plane only
            latest_debug_frame.update(y, [], time.time_ns())
        except Exception:
            logger.exception("Debug frame callback error")

    def listener_callback_hbm(self, msg: Any):
        # During replay the replay thread feeds frames through process_frame()
        # directly — skip the live camera callback to avoid pipeline contention.
        if replaying_active.is_set():
            return
        try:
            fps_counter.tick()
            w, h = msg.width, msg.height
            stride = msg.step if msg.step > 0 else w

            # Zero-copy fast path: when stride == width the NV12 bytes are packed
            # contiguously, so np.frombuffer gives a read-only view and reshape
            # returns another view — no allocation at all.
            if stride == w:
                data_size = h * w * 3 // 2
                nv12 = np.frombuffer(msg.data, dtype=np.uint8)[:data_size].reshape(h * 3 // 2, w)
            else:
                # Stride-padded layout — reconstruct a contiguous NV12 array.
                raw_buffer = np.frombuffer(msg.data, dtype=np.uint8) if not isinstance(msg.data, np.ndarray) else msg.data
                y_plane = raw_buffer[: h * stride].reshape(h, stride)[:, :w]
                uv_start = h * stride
                uv_plane = raw_buffer[uv_start : uv_start + (h // 2) * stride].reshape(h // 2, stride)[:, :w]
                nv12 = np.vstack([y_plane, uv_plane])

            process_frame(nv12_frame=nv12, frame_h=h)

        except Exception as e:
            logger.exception("HBM error")
            self.get_logger().error(f"HBM Fix Error: {e}")
            self.get_logger().error(traceback.format_exc())  # ADD THIS

    def on_inference_done(self, future: Future, tile_x: int, tile_y: int):
        """Handle AI results and push to Django's output_buffer."""
        # global max_output_timestamp
        try:
            if future in active_futures:
                active_futures.remove(future)

            # 1. Get AI Detections
            # results is an InferenceOutput(worker_pid, timestamp, avg_duration, detections)
            result = future.result()

            # 2. Map coordinates back to the full stereo image
            mapped_detections = []
            for det in result.detections:
                # det.bbox is [x1, y1, x2, y2] relative to the 640x640 tile
                # We add the tile offset to get global coordinates
                x1 = det.bbox[0] + tile_x
                y1 = det.bbox[1] + tile_y
                x2 = det.bbox[2] + tile_x
                y2 = det.bbox[3] + tile_y

                # Convert back to [x, y, w, h] for your existing drawing logic
                bbox_global = (x1, y1, x2 - x1, y2 - y1)

                mapped_detections.append(Detection(label=det.label, confidence=det.confidence, bbox=bbox_global))

            # 3. Push to output_buffer for the Django Ninja API to stream
            # We need a 'frame_lores' for the UI.
            # (In tiling mode, we might just pass a dummy or the last processed tile)
            # output_buffer.append(
            #     OutputResult(
            #         worker_pid=result.worker_pid,
            #         timestamp=result.timestamp,
            #         frame_lores=None,  # You'll need to handle preview frame separately
            #         detections_denormalized=mapped_detections,
            #     ),
            # )

        except Exception as e:
            self.get_logger().error(f"Inference Result Error: {e}")

    def publish_detections(self, detections: list):
        """Takes a list of Detection namedtuples, converts them to JSON, and publishes them to the ROS 2 topic."""
        # Format detections for JSON serialization
        # detections is a list of Detection(label, confidence, bbox)
        data = [
            {
                "label": d.label,
                "confidence": float(d.confidence),
                "bbox": [float(x) for x in d.bbox],  # Ensure coordinates are floats
            }
            for d in detections
        ]

        if not self.context.ok():
            return
        msg = String()
        msg.data = json.dumps(data)
        self.target_pub.publish(msg)
        # self.get_logger().info(f"Published {len(data)} detections")

# Single inference thread — the BPU is one hardware block and serialises requests
# internally anyway. One caller avoids IPC overhead, pickle costs, and shared
# memory round-trips that the ProcessPoolExecutor required.
inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piki-inference")
thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piki-streamer")
ros_node: Optional["PikiVisionNode"] = None


# worker_slot_semaphore = Semaphore(NUM_AI_WORKERS)


def frame_producer(ffmpeg_stdout: IO[bytes], buffer_instance: DoubleBuffer):
    """Reads raw frames from FFmpeg's stdout pipe and writes them into a DoubleBuffer."""
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
max_output_timestamp = 0
dashboard = LiveMetricsDashboard()


# try:
#     import picamera2
#     from picamera2.encoders import MJPEGEncoder
#     from picamera2.outputs import FileOutput
#     from libcamera import controls
#
#     PICAMERA_AVAILABLE = True
# except ImportError:
#     logger.info("Picamera2 not available", traceback.format_exc())
#     PICAMERA_AVAILABLE = False
#     picamera2 = None
#     MJPEGEncoder = None
#     FileOutput = None
#     controls = None


def get_measure(description: str):
    now = time.perf_counter()

    def measure(*, log: bool = True) -> None:
        end = time.perf_counter()
        ms = str(round((end - now) * 1000, 2)).rjust(5)
        if log:
            logger.info(f"{description}: {ms} ms!")

    return measure


def _compute_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two normalized [ymin, xmin, ymax, xmax] boxes."""
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


def _nms_detections(
    detections: "list[Detection]",
    iou_threshold: float = 0.45,
    containment_threshold: float = 0.6,
) -> "list[Detection]":
    """Remove duplicate/overlapping detections after cross-tile aggregation.

    Two suppression criteria are applied — a detection is suppressed if it
    overlaps an already-kept higher-confidence detection by either:

    * IoU ≥ ``iou_threshold``  (standard NMS — catches near-identical boxes)
    * IoM ≥ ``containment_threshold``  (Intersection-over-Minimum — catches the
      common tile-boundary case where the same object is fully detected in one tile
      but only partially clipped in the adjacent tile, yielding a small box that
      is almost entirely contained inside the larger one, yet their IoU is low)

    Per-class first pass, then a cross-class containment pass so that a box of one
    class that is almost entirely inside a box of another class is also removed.
    """
    if len(detections) <= 1:
        return detections

    def _overlap(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
        """Return (iou, iom) for two [ymin,xmin,ymax,xmax] boxes."""
        inter_h = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        inter_w = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = inter_h * inter_w
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        iou = inter / union if union > 0 else 0.0
        iom = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
        return iou, iom

    def _greedy_nms(indices: list[int]) -> list[int]:
        """Greedy NMS: keep highest-confidence box, suppress overlapping ones."""
        sorted_by_conf = sorted(indices, key=lambda i: detections[i].confidence, reverse=True)
        suppressed: set[int] = set()
        kept: list[int] = []
        for i in sorted_by_conf:
            if i in suppressed:
                continue
            kept.append(i)
            for j in sorted_by_conf:
                if j == i or j in suppressed:
                    continue
                iou, iom = _overlap(detections[i].bbox, detections[j].bbox)
                if iou >= iou_threshold or iom >= containment_threshold:
                    suppressed.add(j)
        return kept

    # Pass 1: per-class NMS — removes same-label duplicates (tile boundary artefacts,
    # multiple grid cells firing on the same object, etc.)
    from collections import defaultdict  # noqa: PLC0415
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, det in enumerate(detections):
        by_label[det.label].append(i)

    after_per_class: list[int] = []
    for indices in by_label.values():
        if len(indices) == 1:
            after_per_class.extend(indices)
        else:
            after_per_class.extend(_greedy_nms(indices))

    if len(after_per_class) <= 1:
        kept_set = set(after_per_class)
        return [det for i, det in enumerate(detections) if i in kept_set]

    # Pass 2: cross-class containment — removes a detection of one class that is
    # almost entirely inside a detection of a different class (IoM only, not IoU,
    # so legitimate separate objects at similar positions are kept).
    sorted_all = sorted(after_per_class, key=lambda i: detections[i].confidence, reverse=True)
    suppressed: set[int] = set()
    for idx, i in enumerate(sorted_all):
        if i in suppressed:
            continue
        for j in sorted_all[idx + 1:]:
            if j in suppressed:
                continue
            _, iom = _overlap(detections[i].bbox, detections[j].bbox)
            if iom >= containment_threshold:
                suppressed.add(j)

    final_kept = set(i for i in after_per_class if i not in suppressed)
    return [det for i, det in enumerate(detections) if i in final_kept]


def run_object_detection(
    frame_hires: np.ndarray,
    rois: list[Box],
    timestamp: int,
) -> InferenceOutput:
    """Run tile-based inference on the hi-res frame.

    Now runs in a single background thread (inference_pool) rather than a
    ProcessPoolExecutor. Benefits:
      - No pickle/IPC overhead — frame passed by reference within the process
      - No shared memory round-trip
      - BPU gets one sequential caller instead of 3 competing processes
    """
    worker_pid = mp.current_process().pid or 0
    if is_object_detection_disabled.is_set():
        return InferenceOutput(worker_pid=worker_pid, timestamp=timestamp, avg_duration=0, detections=[])

    _profile = bool(os.environ.get("PIKI_PROFILE"))

    try:
        frame_w = frame_hires.shape[1]
        frame_h = frame_hires.shape[0] * 2 // 3  # NV12: total rows = h * 1.5

        from .ai import MODEL_INPUT_TYPE, detect_objects  # noqa: PLC0415

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
        logger.debug("Tiles to infer: %d", len(tiles))

        total_duration = 0
        all_detections: list[Detection] = []

        for tile_img, tile_x, tile_y in tiles:
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

        # Collect tile origins and boundaries for edge-proximity check.
        seen_origins: set[tuple[int, int]] = set()
        tile_edges: list[tuple[int, int, int, int]] = []  # (left, top, right, bottom)
        for _, tx, ty in tiles:
            seen_origins.add((tx, ty))
            tile_edges.append((tx, ty, tx + ai_input_size, ty + ai_input_size))

        _t = time.perf_counter()
        all_detections = _nms_detections(all_detections, iou_threshold=0.45)
        if _profile:
            logger.info("PERF stage=cross_tile_nms ms=%.2f", (time.perf_counter() - _t) * 1000)

        # --- Second pass: re-detect objects that touch a tile boundary ----------
        EDGE_MARGIN = 32  # pixels — ~5 % of a 640 px tile

        extra_tiles: list[tuple[int, int]] = []
        for det in all_detections:
            x1 = int(det.bbox[1] * frame_w)
            y1 = int(det.bbox[0] * frame_h)
            x2 = int(det.bbox[3] * frame_w)
            y2 = int(det.bbox[2] * frame_h)

            for left, top, right, bottom in tile_edges:
                if (abs(x1 - left) <= EDGE_MARGIN
                        or abs(x2 - right) <= EDGE_MARGIN
                        or abs(y1 - top) <= EDGE_MARGIN
                        or abs(y2 - bottom) <= EDGE_MARGIN):
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    new_tx = max(0, min(cx - ai_input_size // 2, frame_w - ai_input_size))
                    new_ty = max(0, min(cy - ai_input_size // 2, frame_h - ai_input_size))
                    if (new_tx, new_ty) not in seen_origins:
                        extra_tiles.append((new_tx, new_ty))
                        seen_origins.add((new_tx, new_ty))
                    break  # one matching edge is enough

        if extra_tiles:
            from .func import _slice_nv12_tile  # noqa: PLC0415

            for tx, ty in extra_tiles:
                tile_img = _slice_nv12_tile(
                    nv12=frame_hires, buffer_h=frame_h, tx=tx, ty=ty, tile_size=ai_input_size,
                )
                duration, detections = detect_objects(tile_img)
                total_duration += duration

                for label, confidence, local_pixel_bbox in detections:
                    lx1, ly1, lx2, ly2 = local_pixel_bbox
                    all_detections.append(Detection(
                        label=label,
                        confidence=confidence,
                        bbox=[
                            (ly1 + ty) / frame_h,
                            (lx1 + tx) / frame_w,
                            (ly2 + ty) / frame_h,
                            (lx2 + tx) / frame_w,
                        ],
                    ))

            all_detections = _nms_detections(all_detections, iou_threshold=0.45)

        # -----------------------------------------------------------------------

        avg_duration = 0 if not tiles else total_duration // (len(tiles) + len(extra_tiles))
        return InferenceOutput(
            worker_pid=worker_pid,
            timestamp=timestamp,
            avg_duration=avg_duration,
            detections=all_detections,
        )

    except:
        logger.exception("Fatal error in inference thread (PID: %d)", os.getpid())
        raise


def denormalize(*, bbox_normalized: Sequence[int], frame_shape: Sequence[int]) -> Box:
    frame_height, frame_width = frame_shape
    ymin, xmin, ymax, xmax = bbox_normalized

    ymin = max(0.0, ymin)
    xmin = max(0.0, xmin)
    ymax = min(1.0, ymax)
    xmax = min(1.0, xmax)

    left = int(xmin * frame_width)
    top = int(ymin * frame_height)
    right = int(xmax * frame_width)
    bottom = int(ymax * frame_height)
    width = right - left
    height = bottom - top
    return Box(left, top, width, height)


# def denormalize_detections(detections: list[OutputResult], frame_shape) -> list[Detection]:
#     detections_denormalized: list[Detection] = []
#     for result in detections:
#         x, y, w, h = denormalize(bbox_normalized, frame_shape)
#
#         detections_denormalized.append(Detection(label=label, confidence=confidence, bbox=(x, y, w, h)))
#     return detections_denormalized


def on_done(future: Future[InferenceOutput]):
    """Handle completed inference."""
    global max_output_timestamp, _locked_target_bbox, _locked_target_label, _locked_target_lost_since
    active_futures.remove(future)
    try:
        worker_pid, timestamp, inference_time, detections = future.result()

        if timestamp < max_output_timestamp:
            logger.info(f"Inference result arrived late, discarding (timestamp={timestamp})")
            with cache_lock:
                lowres_frame_cache.pop(timestamp, None)
            return

        max_output_timestamp = timestamp

        aim_enabled = app_settings.aim_settings.servo_enabled
        target_classes = {c.strip().lower() for c in (app_settings.aim_settings.target_classes or [])}
        target_lock_duration = float(app_settings.aim_settings.target_lock_duration)

        # --- Servo aiming + tracker init/re-init from YOLO result ---
        # Always process regardless of streaming state so servo and tracker work
        # even when no browser is watching.
        #
        # Target-lock: instead of blindly picking the first matching detection each
        # frame (which causes the servo to jump when detection order changes), we
        # maintain a locked target bbox.  We look for the matching detection with the
        # highest IoU vs. the lock.  If found, we update the lock and aim.  If not
        # found for longer than `target_lock_duration` seconds, we allow a switch.

        # Collect all matching detections for this frame.
        matching: list[tuple[str, float, list[float]]] = [
            (label, confidence, bbox_normalized)
            for label, confidence, bbox_normalized in detections
            if not target_classes or label.strip().lower() in target_classes
        ]

        # Resolve which detection to use this frame.
        chosen: tuple[str, float, list[float]] | None = None

        if matching:
            if _locked_target_bbox is None:
                # No lock yet — take the first match and lock onto it.
                chosen = matching[0]
            else:
                # Find the matching detection with highest IoU vs. current lock.
                best_iou = 0.0
                best = None
                for det in matching:
                    iou = _compute_iou(_locked_target_bbox, det[2])
                    if iou > best_iou:
                        best_iou = iou
                        best = det
                if best_iou >= 0.3:
                    # Good match — update lock position, clear lost timer.
                    chosen = best
                    _locked_target_lost_since = None
                else:
                    # Current lock not found among detections.
                    if _locked_target_lost_since is None:
                        _locked_target_lost_since = time.time()
                    elapsed = time.time() - _locked_target_lost_since
                    if elapsed >= target_lock_duration:
                        # Lock expired — switch to the new first matching detection.
                        logger.info(
                            "Target lock expired after %.1fs, switching to new target.", elapsed
                        )
                        chosen = matching[0]
                        _locked_target_lost_since = None
                    # else: hold lock position; skip servo update this cycle.
        else:
            # No matching detections at all.
            if _locked_target_bbox is not None:
                if _locked_target_lost_since is None:
                    _locked_target_lost_since = time.time()
                elapsed = time.time() - _locked_target_lost_since
                if elapsed >= target_lock_duration:
                    logger.info("Target lock expired after %.1fs (no detections), clearing lock.", elapsed)
                    _locked_target_bbox = None
                    _locked_target_label = None
                    _locked_target_lost_since = None

        if chosen is not None:
            chosen_label, chosen_confidence, chosen_bbox = chosen
            _locked_target_bbox = list(chosen_bbox)
            _locked_target_label = chosen_label

            # Aim servo at chosen detection.
            if aim_enabled and target_classes:
                from .engine import aim_at  # noqa: PLC0415
                _aim_center = None
                if _latest_mask is not None:
                    _aim_center = _foreground_centroid(chosen_bbox, _latest_mask, _latest_mask_shape)
                aim_at(bbox_normalized=chosen_bbox, aim_center=_aim_center)

        dashboard.update(worker_id=worker_pid, inference_time=inference_time)

        # --- Display work: only when a stream consumer is active ---
        if streaming_active.is_set():
            with cache_lock:
                lores_shape = lowres_frame_cache.pop(timestamp, None)

            detections_denormalized: list[Detection] = []
            for label, confidence, bbox_normalized in detections:
                if lores_shape is None:
                    break
                x, y, w, h = denormalize(bbox_normalized=bbox_normalized, frame_shape=lores_shape)
                if x < 0 or y < 0 or w < 0 or h < 0:
                    logger.warning("Abnormal denormalized bbox: %s → %s", bbox_normalized, (x, y, w, h))
                    continue
                detections_denormalized.append(
                    Detection(label=label, confidence=confidence, bbox=(x, y, w, h)),
                )

            if ros_node is not None:
                ros_node.publish_detections(detections_denormalized)

            with latest_ai_lock:
                global latest_ai_detections
                latest_ai_detections = detections_denormalized
        else:
            with cache_lock:
                lowres_frame_cache.pop(timestamp, None)

            if ros_node is not None:
                ros_node.publish_detections([
                    Detection(label=label, confidence=confidence, bbox=list(bbox))
                    for label, confidence, bbox in detections
                ])

        # --- Event-triggered recording: check if any detection matches trigger classes ---
        if (event_recording_enabled.is_set()
                and not event_recording_active.is_set()
                and not recording_active.is_set()
                and time.time() > event_recording_cooldown_until):
            with event_trigger_classes_lock:
                trigger_set = {c.strip().lower() for c in event_trigger_classes}
            if trigger_set:
                for label, confidence, bbox_normalized in detections:
                    if confidence >= prob_threshold.value and label.strip().lower() in trigger_set:
                        _start_event_recording()
                        break

    except KeyboardInterrupt:
        logger.info("Shutting down on KeyboardInterrupt in on_done.")
    except:
        traceback.print_exc()
        raise


def _start_event_recording() -> None:
    """Begin event-triggered recording: flush pre-buffer, start live capture."""
    global _event_recorder, _event_clip_until
    from datetime import datetime  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from django.conf import settings as django_settings  # noqa: PLC0415

    from .recording import EventClipRecorder, pre_buffer_snapshot  # noqa: PLC0415

    pre_frames = pre_buffer_snapshot()

    if pre_frames:
        h, w = pre_frames[0].shape[:2]
    else:
        h, w = 360, 640

    videos_dir = Path(django_settings.MEDIA_ROOT) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(videos_dir / f"event_{ts}.mp4")

    with _event_recorder_lock:
        _event_recorder = EventClipRecorder(path, fps=30.0, pre_frames=pre_frames,
                                             frame_w=w, frame_h=h)
        _event_clip_until = time.monotonic() + event_post_trigger_seconds.value
        event_recording_active.set()

    logger.info("Event recording started: %s (%d pre-buffer frames, %ds post-trigger)",
                path, len(pre_frames), event_post_trigger_seconds.value)


def _finalize_event_recording() -> None:
    """Stop event recording, save to DB, queue notification."""
    global _event_recorder, _event_clip_until
    from datetime import datetime  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from django.conf import settings as django_settings  # noqa: PLC0415

    with _event_recorder_lock:
        if _event_recorder is None:
            event_recording_active.clear()
            return
        final_path, frame_count, error = _event_recorder.close()
        _event_recorder = None

    _event_clip_until = 0.0
    event_recording_active.clear()
    from . import shared as _s  # noqa: PLC0415
    _s.event_recording_cooldown_until = time.time() + event_cooldown_seconds.value

    if final_path and frame_count > 0:
        file_path = Path(final_path)
        try:
            from ..models import Video  # noqa: PLC0415

            video = Video.objects.create(
                filename=file_path.name,
                file=str(file_path.relative_to(django_settings.MEDIA_ROOT)),
                size_bytes=file_path.stat().st_size,
                source="event",
            )
            with event_clip_queue_lock:
                event_clip_queue.append({
                    "filename": file_path.name,
                    "file": str(file_path.relative_to(django_settings.MEDIA_ROOT)),
                    "frame_count": frame_count,
                    "time": datetime.now().isoformat(),
                    "video_id": video.id,
                })
            logger.info("Event clip saved: %s (%d frames)", file_path.name, frame_count)
        except Exception:
            logger.exception("Failed to save event clip to DB")
    else:
        logger.warning("Event recording produced no output: error=%s", error)


def _submit_yolo(*, nv12_frame: np.ndarray, frame_lores: np.ndarray, rois: list, timestamp: int) -> None:
    """Submit a YOLO inference job."""
    if streaming_active.is_set():
        with cache_lock:
            # Store only (h, w) — denormalize() unpacks as (height, width) and
            # would fail if we stored a 3-tuple for BGR frames.
            lowres_frame_cache[timestamp] = frame_lores.shape[:2]
    future = inference_pool.submit(run_object_detection, frame_hires=nv12_frame, rois=rois, timestamp=timestamp)
    active_futures.append(future)
    future.add_done_callback(on_done)


def process_frame(*, nv12_frame: np.ndarray, frame_h: int):
    global latest_ai_detections
    global _latest_mask, _latest_mask_shape

    current_time = time.time_ns()

    # Downscale for motion detection and preview.
    # Zero-copy stride-2 decimation (view into nv12_frame) — ~80x faster than cv2.resize
    # for the current 640x352 input. OpenCV MOG2 handles non-contiguous arrays natively.
    y_plane = nv12_frame[:frame_h]
    step = preview_downscale_factor
    frame_lores = y_plane[::step, ::step]
    has_movement, mask = motion_detector.is_moving(frame_lores)
    _latest_mask = mask
    _latest_mask_shape = mask.shape[:2] if mask is not None else (1, 1)

    # --- Recording: manual and event-triggered ---
    frame_bgr = cv2.cvtColor(frame_lores, cv2.COLOR_GRAY2BGR)

    # Pre-buffer: always append when event recording is enabled.
    if event_recording_enabled.is_set():
        from .recording import pre_buffer_append  # noqa: PLC0415

        pre_buffer_append(frame_bgr, event_pre_buffer_seconds.value)

    # Manual recording.
    if recording_active.is_set():
        from .recording import write_frame  # noqa: PLC0415

        write_frame(frame_bgr)

    # Event-triggered recording (post-trigger live capture).
    if event_recording_active.is_set():
        with _event_recorder_lock:
            if _event_recorder is not None:
                _event_recorder.write_frame(frame_bgr)
        # Check if post-trigger duration has elapsed.
        if time.monotonic() >= _event_clip_until:
            _finalize_event_recording()

    detections_to_show = []

    if os.environ.get("DISABLE_AI"):
        # DISABLE_AI: skip YOLO entirely, only draw debug overlays.
        if app_settings.debug_settings.show_mask or app_settings.debug_settings.show_rois:
            gray = frame_lores if frame_lores.ndim == 2 else cv2.cvtColor(frame_lores, cv2.COLOR_BGR2GRAY)
            frame_lores = cv2.merge((gray, gray, gray))

            if app_settings.debug_settings.show_rois:
                rois = motion_detector.create_rois(mask=mask)
                for roi in rois:
                    rx, ry, rw, rh = roi
                    cv2.rectangle(frame_lores, (rx, ry), (rx + rw, ry + rh), (0, 200, 255), 2)

            if app_settings.debug_settings.show_mask:
                frame_lores = motion_detector.highlight_movement_on(
                    frame=frame_lores,
                    mask=mask,
                    overlay_color_rgb=(147, 20, 255),
                    transparency_factor=mask_transparency.value,
                    draw_boxes=True,
                )
    elif has_movement:
        try:
            if not active_futures:
                rois = motion_detector.create_rois(mask=mask)
                if rois:
                    _submit_yolo(nv12_frame=nv12_frame, frame_lores=frame_lores, rois=rois, timestamp=time.monotonic_ns())
        except Exception:
            logger.exception("Error in process_frame AI logic")

        if streaming_active.is_set():
            with latest_ai_lock:
                detections_to_show = latest_ai_detections

        # Draw mask / ROIs overlays independently of YOLO.
        if app_settings.debug_settings.show_mask or app_settings.debug_settings.show_rois:
            # frame_lores may be the 2-D Y-plane; make it a writable BGR copy.
            if frame_lores.ndim == 2:
                frame_lores = cv2.merge((frame_lores, frame_lores, frame_lores))
            else:
                frame_lores = frame_lores.copy()

            if app_settings.debug_settings.show_rois:
                # Re-create ROIs (same as what was submitted to YOLO above).
                rois = motion_detector.create_rois(mask=mask)
                for roi in rois:
                    rx, ry, rw, rh = roi
                    cv2.rectangle(frame_lores, (rx, ry), (rx + rw, ry + rh), (0, 200, 255), 2)

            if app_settings.debug_settings.show_mask:
                frame_lores = motion_detector.highlight_movement_on(
                    frame=frame_lores,
                    mask=mask,
                    overlay_color_rgb=(147, 20, 255),
                    transparency_factor=mask_transparency.value,
                    draw_boxes=True,
                )
    else:
        if streaming_active.is_set():
            # No movement — clear stale detections for the display.
            with latest_ai_lock:
                latest_ai_detections = []
            detections_to_show = []

    # Update the shared latest_frame state only while a client is watching.
    if streaming_active.is_set():
        latest_frame.update(frame_lores.copy(), detections_to_show, current_time)


def stream_nonblocking():
    thread_pool.submit(stream_with_ros)


def stream_with_ros():
    try:
        global ros_node

        # from cv_bridge import CvBridge

        delay_seconds = 1
        logger.info(f"Starting ROS 2 videostream in {delay_seconds}s...")
        time.sleep(delay_seconds)

        print("------------!!!!!!!!!!!!! STREAM (ROS 2)")
        # high_res_w, high_res_h = 640, 352

        rclpy.init(args=None)
        ros_node = PikiVisionNode()
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

    # input_buffer.close()

    thread_pool.shutdown(wait=True, cancel_futures=True)
    inference_pool.shutdown(wait=True, cancel_futures=True)
    if ffmpeg_process:
        ffmpeg_process.kill()

    logger.info("[DJANGO SHUTDOWN] Processes stopped..")
