import atexit
import functools
import json
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import threading
import time
import traceback
from collections import deque
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
    fps_counter,
    is_object_detection_disabled,
    latest_debug_frame,
    latest_frame,
    mask_transparency,
    motion_detector,
    preview_downscale_factor,
    streaming_active,
    tracker_active,
    tracker_lost_threshold,
    tracking_enabled,
)

logger = logging.getLogger(__name__)


last_known_bbox = None
last_known_velocity = None
untracked_frames_count = 0
total_untracked_frames_count = 0
velocity_buffer = deque(maxlen=15)

# NOTE: last_inference_frame was removed. The lores frame is now bound directly
# to each inference future via functools.partial inside _submit_yolo, eliminating
# a race where a new _submit_yolo call could overwrite it before on_done reads it.

# Re-run YOLO every N frames while tracking to correct any drift.
_YOLO_REVALIDATE_INTERVAL = 30
_yolo_revalidate_counter = 0

# CSRT/KCF can report found=False on a single low-contrast frame even when the
# object is still present.  Require this many consecutive failures before we
# actually reset the tracker so transient drops don't kill a live track.
# The threshold is read at runtime from the shared tracker_lost_threshold mp.Value
# so it can be adjusted from the web UI without a restart.
_tracker_lost_streak = 0

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

tracker_lock = Lock()
tracking = mp.Event()
coasting = mp.Event()
tracker: cv2.Tracker | None = None

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

        _t = time.perf_counter()
        all_detections = _nms_detections(all_detections, iou_threshold=0.45)
        if _profile:
            logger.info("PERF stage=cross_tile_nms ms=%.2f", (time.perf_counter() - _t) * 1000)

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


def _make_tracker() -> cv2.Tracker:
    """Create a tracker instance based on the current tracker_type setting."""
    from .shared import get_tracker_type as _get_tracker_type  # noqa: PLC0415
    if _get_tracker_type() == "KCF":
        return cv2.TrackerKCF.create()
    return cv2.TrackerCSRT.create()


def get_tracker_type() -> str:
    from .shared import get_tracker_type as _get  # noqa: PLC0415
    return _get()


def on_done(future: Future[InferenceOutput], lores_frame: "np.ndarray | None" = None):
    """Handle completed inference.

    ``lores_frame`` is the contiguous grayscale lores frame captured at submission
    time (bound via ``functools.partial`` in ``_submit_yolo``).  It must never be
    read from a shared global here because ``active_futures.remove(future)`` (the
    first thing this function does) opens a window where the main thread can submit
    a new YOLO job and overwrite any shared reference before we reach the tracker
    init code.
    """
    global max_output_timestamp, tracker, untracked_frames_count, total_untracked_frames_count
    global _tracker_lost_streak, _locked_target_bbox, _locked_target_label, _locked_target_lost_since
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
                aim_at(bbox_normalized=chosen_bbox)

            # Re-init tracker with this YOLO detection (corrects any drift).
            # Use the lores_frame bound at submission time — reading any shared
            # global here would race with _submit_yolo on the main thread.
            if lores_frame is not None and tracking_enabled.is_set():
                fh, fw = lores_frame.shape[:2]
                x, y, w, h = denormalize(bbox_normalized=chosen_bbox, frame_shape=(fh, fw))
                # Clamp to frame bounds: denormalize() may produce right/bottom == fw/fh
                # when xmax/ymax == 1.0, which OpenCV trackers reject.
                w = min(w, fw - x)
                h = min(h, fh - y)
                if w > 0 and h > 0:
                    # Keep grayscale for init: the camera outputs a Y-plane, so
                    # converting to BGR produces 3 identical channels which breaks
                    # KCF's colour-name features.  Grayscale works for both trackers.
                    frame_for_tracker = np.ascontiguousarray(lores_frame)
                    if frame_for_tracker.ndim == 3:
                        frame_for_tracker = cv2.cvtColor(frame_for_tracker, cv2.COLOR_BGR2GRAY)
                    new_tracker = _make_tracker()
                    try:
                        # OpenCV 4.x init() returns None (void); older versions return
                        # bool True/False.  Assume success unless an exception is raised.
                        new_tracker.init(frame_for_tracker, (x, y, w, h))
                    except cv2.error as e:
                        logger.warning("Tracker init raised cv2.error (%s), skipping.", e)
                    else:
                        with tracker_lock:
                            tracker = new_tracker
                            tracking.set()
                            untracked_frames_count = 0
                            total_untracked_frames_count = 0
                        _tracker_lost_streak = 0
                        tracker_active.set()
                        logger.info("Tracker (re-)initialised at bbox (%d,%d,%d,%d) frame=%s type=%s",
                                    x, y, w, h, lores_frame.shape, get_tracker_type())

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

    except KeyboardInterrupt:
        logger.info("Shutting down on KeyboardInterrupt in on_done.")
    except:
        traceback.print_exc()
        raise


def _submit_yolo(*, nv12_frame: np.ndarray, frame_lores: np.ndarray, rois: list, timestamp: int) -> None:
    """Submit a YOLO inference job and bind the lores frame to the done callback.

    The lores snapshot is passed directly to ``on_done`` via ``functools.partial``
    so it is guaranteed to be the frame from *this* specific submission, with no
    shared-global race.  ``np.ascontiguousarray`` creates an independent copy when
    ``frame_lores`` is a strided view; for an already-contiguous array we call
    ``.copy()`` explicitly to ensure the snapshot is never an alias.
    """
    lores_snapshot = np.ascontiguousarray(frame_lores)
    if lores_snapshot is frame_lores:
        # frame_lores was already contiguous — ascontiguousarray returned the same
        # object.  Make an independent copy so the snapshot outlives the caller.
        lores_snapshot = lores_snapshot.copy()
    if streaming_active.is_set():
        with cache_lock:
            # Store only (h, w) — denormalize() unpacks as (height, width) and
            # would fail if we stored a 3-tuple for BGR frames.
            lowres_frame_cache[timestamp] = lores_snapshot.shape[:2]
    future = inference_pool.submit(run_object_detection, frame_hires=nv12_frame, rois=rois, timestamp=timestamp)
    active_futures.append(future)
    future.add_done_callback(functools.partial(on_done, lores_frame=lores_snapshot))


def process_frame(*, nv12_frame: np.ndarray, frame_h: int):
    global tracker, untracked_frames_count, total_untracked_frames_count
    global last_known_bbox, latest_ai_detections, _yolo_revalidate_counter, _tracker_lost_streak

    current_time = time.time_ns()

    # Downscale for motion detection and preview.
    # Zero-copy stride-2 decimation (view into nv12_frame) — ~80x faster than cv2.resize
    # for the current 640x352 input. OpenCV MOG2 handles non-contiguous arrays natively.
    y_plane = nv12_frame[:frame_h]
    step = preview_downscale_factor
    frame_lores = y_plane[::step, ::step]
    has_movement, mask = motion_detector.is_moving(frame_lores)

    with tracker_lock:
        is_tracking = tracking.is_set()

    # If tracking has been disabled externally, stop any active tracker now.
    if is_tracking and not tracking_enabled.is_set():
        with tracker_lock:
            tracker = None
            tracking.clear()
            untracked_frames_count = 0
            total_untracked_frames_count = 0
        tracker_active.clear()
        _tracker_lost_streak = 0
        _yolo_revalidate_counter = 0
        is_tracking = False

    detections_to_show = []

    if is_tracking:
        assert tracker
        # Pass the raw grayscale Y-plane directly — converting to BGR produces
        # 3 identical channels which breaks KCF's colour-name features.
        contiguous = np.ascontiguousarray(frame_lores)
        if contiguous.ndim == 3:
            contiguous = cv2.cvtColor(contiguous, cv2.COLOR_BGR2GRAY)
        found = False
        try:
            found, raw_bbox = tracker.update(contiguous)
        except cv2.error as e:
            # cv2.error means the tracker's internal state is corrupt (e.g. ROI
            # drifted off-frame).  This is not a transient "briefly lost" event —
            # reset immediately rather than counting toward the lost streak.
            logger.warning("Tracker update raised cv2.error (%s), resetting immediately.", e)
            with tracker_lock:
                tracker = None
                tracking.clear()
                untracked_frames_count = 0
                total_untracked_frames_count = 0
            tracker_active.clear()
            _tracker_lost_streak = 0
            _yolo_revalidate_counter = 0
            is_tracking = False

        if is_tracking and found:
            _tracker_lost_streak = 0
            x, y, w, h = (int(v) for v in raw_bbox)
            last_known_bbox = (x, y, w, h)
            detections_to_show = [Detection(label="tracker", confidence=1.0, bbox=(x, y, w, h))]

            # Aim servo from tracker position every frame (smooth aiming between YOLO calls).
            aim_enabled = app_settings.aim_settings.servo_enabled
            target_classes = {c.strip().lower() for c in (app_settings.aim_settings.target_classes or [])}
            if aim_enabled and target_classes:
                fh, fw = contiguous.shape[:2]
                bbox_norm = [y / fh, x / fw, (y + h) / fh, (x + w) / fw]
                from .engine import aim_at  # noqa: PLC0415
                aim_at(bbox_normalized=bbox_norm)

            if ros_node is not None:
                ros_node.publish_detections(detections_to_show)

            # Periodically re-run YOLO to correct tracker drift.
            _yolo_revalidate_counter += 1
            if _yolo_revalidate_counter >= _YOLO_REVALIDATE_INTERVAL and not active_futures:
                _yolo_revalidate_counter = 0
                # Only consider motion that falls *outside* the tracked bbox — motion
                # inside the bbox is the tracked object itself, not a reason to re-run YOLO.
                outside_mask = mask.copy()
                outside_mask[y : y + h, x : x + w] = 0
                _px_threshold = settings.foreground_mask_options.pixelcount_threshold.value
                has_outside_motion = cv2.countNonZero(outside_mask) >= _px_threshold
                rois = motion_detector.create_rois(mask=outside_mask) if has_outside_motion else []
                if not rois:
                    # No motion outside bbox (object is still or all motion is inside the
                    # bbox) — always revalidate using the tracker's current bbox so YOLO
                    # can confirm the object is still there and correct any drift.
                    from .interfaces import Box as _Box  # noqa: PLC0415
                    rois = [_Box(x, y, w, h)]
                try:
                    _submit_yolo(nv12_frame=nv12_frame, frame_lores=contiguous, rois=rois, timestamp=time.monotonic_ns())
                except Exception:
                    logger.exception("Error submitting YOLO revalidation during tracking")
        elif is_tracking:
            # Tracker returned found=False — count consecutive failures before committing
            # to a reset.  A single low-contrast frame can produce a false negative.
            _tracker_lost_streak += 1
            if _tracker_lost_streak < tracker_lost_threshold.value:
                # Keep the last known position visible while we wait for confirmation.
                if last_known_bbox is not None:
                    lx, ly, lw, lh = last_known_bbox
                    detections_to_show = [Detection(label="tracker", confidence=0.5, bbox=(lx, ly, lw, lh))]
            else:
                # Confirmed loss — reset and let motion detection take over.
                logger.info("Tracker lost target after %d consecutive failures, resetting.", _tracker_lost_streak)
                _tracker_lost_streak = 0
                with tracker_lock:
                    tracker = None
                    tracking.clear()
                    untracked_frames_count = 0
                    total_untracked_frames_count = 0
                tracker_active.clear()
                _yolo_revalidate_counter = 0
                # Immediately try motion-based YOLO on this same frame.
                if has_movement and not active_futures:
                    rois = motion_detector.create_rois(mask=mask)
                    if rois:
                        try:
                            _submit_yolo(nv12_frame=nv12_frame, frame_lores=frame_lores, rois=rois, timestamp=time.monotonic_ns())
                        except RuntimeError:
                            pass  # thread pool shut down during exit — ignore
                        except Exception:
                            logger.exception("Error submitting YOLO after tracker loss")
                if streaming_active.is_set():
                    with latest_ai_lock:
                        detections_to_show = latest_ai_detections

    elif app_settings.debug_settings.debug_enabled or os.environ.get("DISABLE_AI"):
        mode = app_settings.debug_settings.mode
        gray = frame_lores if frame_lores.ndim == 2 else cv2.cvtColor(frame_lores, cv2.COLOR_BGR2GRAY)
        frame_lores = cv2.merge((gray, gray, gray))

        if mode == "rois":
            # Draw the exact tile rectangles that will be sent to YOLO, so the user
            # can see what motion detection selected.
            rois = motion_detector.create_rois(mask=mask)
            for roi in rois:
                rx, ry, rw, rh = roi
                cv2.rectangle(frame_lores, (rx, ry), (rx + rw, ry + rh), (0, 200, 255), 2)
            frame_lores = motion_detector.highlight_movement_on(
                frame=frame_lores,
                mask=mask,
                overlay_color_rgb=(147, 20, 255),
                transparency_factor=mask_transparency.value,
                draw_boxes=False,
            )
        else:
            # "mask" mode — show motion blobs with bounding boxes
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
