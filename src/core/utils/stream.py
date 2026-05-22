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
from multiprocessing import Lock, Semaphore
from pathlib import Path
from typing import IO, Any, Optional

import norfair
import numpy as np
import rclpy
from django.conf import settings as django_settings
from sensor_msgs.msg import Image as RosImage
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from . import shared as _s
from .ai import MODEL_INPUT_TYPE, detect_objects
from .event_log import EventLogger
from .func import (
    slice_roi_into_tiles,
)
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
    ghost_frames_ms,
    is_object_detection_disabled,
    latest_debug_frame,
    latest_frame,
    mask_transparency,
    min_consecutive_frames,
    motion_detector,
    preview_downscale_factor,
    prob_threshold,
    prob_threshold_keep,
    recording_active,
    servo_aim_confidence,
    replaying_active,
    settings,
    streaming_active,
    tracker_confirm_hits,
    tracker_enabled,
    tracker_iou_threshold,
    tracker_max_misses,
    tracker_reid_enabled,
    tracker_reid_hit_counter_max,
    tracker_reid_threshold,
)

logger = logging.getLogger(__name__)


# Target-lock state: tracks the currently locked detection across YOLO frames so
# the servo doesn't jump when detection order changes or multiple targets exist.
_locked_target_bbox: list[float] | None = None   # normalized [ymin, xmin, ymax, xmax]
_locked_target_label: str | None = None
_locked_target_lost_since: float | None = None   # time.time() when target was last seen
_prev_aim_bbox: list[float] | None = None        # 1-frame delay buffer for servo feed

# --- Phase A stability state (hysteresis, min-streak, ghost frames) ---
_label_streak: dict[str, int] = {}
_last_seen_monotonic_ns: int = 0
_last_seen_denormalized: "list[Detection]" = []

double_buffer: DoubleBuffer | None = None
worker_semaphore = Semaphore(NUM_AI_WORKERS)
ffmpeg_process: subprocess.Popen | None = None
lowres_frame_cache = {}
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

_event_logger: "EventLogger | None" = None
_event_frame_idx: int = 0
_event_started_monotonic: float = 0.0
_event_video_path: str = ""

_latest_inference_log_entries: list[dict] = []
_latest_inference_lock = threading.Lock()

# Module-level Norfair tracker, lazily constructed on first inference.
_tracker: "Optional[norfair.Tracker]" = None
_tracker_params: "Optional[tuple[float, int, int]]" = None
# Guard concurrent _tracker.update() calls — process_frame() (ROS thread) and
# on_done() (inference thread) both call into the tracker.
_tracker_lock = threading.Lock()


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
        debug_topic = os.environ.get("ROS_DEBUG_TOPIC", "/image_right_raw")

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.get_logger().info(f"Subscribing to: {topic_name}")
        self.subscription = self.create_subscription(RosImage, topic_name, self.listener_callback_hbm, qos_profile)
        self.target_pub = self.create_publisher(String, "/piki/detections", 10)

        self.get_logger().info(f"Debug feed subscribing to: {debug_topic}")
        self.create_subscription(RosImage, debug_topic, self._debug_frame_callback, qos_profile)

    def _debug_frame_callback(self, msg: Any):
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
            y = nv12[:h]
            latest_debug_frame.update(y, [], time.time_ns())
        except Exception:
            logger.exception("Debug frame callback error")

    def listener_callback_hbm(self, msg: Any):
        if replaying_active.is_set():
            return
        try:
            fps_counter.tick()
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

    def publish_detections(self, detections: list):
        data = [
            {
                "label": d.label,
                "confidence": float(d.confidence),
                "bbox": [float(x) for x in d.bbox],
            }
            for d in detections
        ]

        if not self.context.ok():
            return
        msg = String()
        msg.data = json.dumps(data)
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


# --- Appearance re-identification (norfair reid) ----------------------------
# Mirrors tryolabs/norfair demos/reid: a per-detection color-histogram embedding
# plus a histogram-correlation distance used by the tracker to re-match lost
# tracks. Embeddings are computed in the worker (where the frame lives) and
# carried to on_done() via InferenceOutput.embeddings.
_REID_HIST_BINS = 128
_REID_PAST_DETECTIONS = 5


def _reid_get_cutout(points: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Crop the image to the [[x1,y1],[x2,y2]] (pixel) box, clamped to bounds."""
    h, w = image.shape[:2]
    min_x = max(0, int(min(points[:, 0])))
    max_x = min(w, int(max(points[:, 0])))
    min_y = max(0, int(min(points[:, 1])))
    max_y = min(h, int(max(points[:, 1])))
    return image[min_y:max_y, min_x:max_x]


def _reid_get_hist(image: np.ndarray) -> "Optional[np.ndarray]":
    """2D U/V color histogram of a crop, normalized — the appearance embedding."""
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return None
    hist = cv2.calcHist(
        [cv2.cvtColor(image, cv2.COLOR_BGR2YUV)],
        [1, 2], None, [_REID_HIST_BINS, _REID_HIST_BINS], [0, 256, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=1.0, beta=0, norm_type=cv2.NORM_MINMAX)
    return hist


def _reid_embedding_distance(matched_not_init_trackers, unmatched_trackers) -> float:
    """norfair reid_distance_function: 1 - histogram correlation, lower = closer.

    Compares the appearance of a recently-lost track against a not-yet-confirmed
    one; returns a small distance only when their histograms correlate well.
    """
    cutoff = float(tracker_reid_threshold.value)
    snd_embedding = unmatched_trackers.last_detection.embedding
    if snd_embedding is None:
        for det in reversed(unmatched_trackers.past_detections):
            if det.embedding is not None:
                snd_embedding = det.embedding
                break
        else:
            return 1.0
    for det_fst in matched_not_init_trackers.past_detections:
        if det_fst.embedding is None:
            continue
        distance = 1.0 - cv2.compareHist(snd_embedding, det_fst.embedding, cv2.HISTCMP_CORREL)
        if distance < cutoff:
            return distance
    return 1.0


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

        # Re-id appearance embeddings (parallel to all_detections). Computed here
        # because the worker is the only place the frame image is available.
        all_embeddings: list = []
        if bool(tracker_reid_enabled.value):
            for det in all_detections:
                d_ymin, d_xmin, d_ymax, d_xmax = det.bbox
                pts = np.array(
                    [[d_xmin * frame_w, d_ymin * frame_h], [d_xmax * frame_w, d_ymax * frame_h]],
                    dtype=np.float32,
                )
                all_embeddings.append(_reid_get_hist(_reid_get_cutout(pts, frame_hires)))

        avg_duration = 0 if not tiles else total_duration // len(tiles)
        return InferenceOutput(
            worker_pid=worker_pid,
            timestamp=timestamp,
            avg_duration=avg_duration,
            detections=all_detections,
            embeddings=all_embeddings,
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


def on_done(future: Future[InferenceOutput]):
    """Handle completed inference."""
    global max_output_timestamp, _locked_target_bbox, _locked_target_label, _locked_target_lost_since, _prev_aim_bbox
    active_futures.remove(future)
    try:
        worker_pid, timestamp, inference_time, detections, embeddings = future.result()

        if timestamp < max_output_timestamp:
            logger.info(f"Inference result arrived late, discarding (timestamp={timestamp})")
            with cache_lock:
                lowres_frame_cache.pop(timestamp, None)
            return

        max_output_timestamp = timestamp

        # Exclusion-zone filter
        from . import exclusion as _exclusion  # noqa: PLC0415

        zones_active = _exclusion.has_zones()
        # Align embeddings (parallel to detections) so they survive the filter.
        if not embeddings or len(embeddings) != len(detections):
            embeddings = [None] * len(detections)
        log_entries: list[dict] = []
        kept: list = []
        kept_embeddings: list = []
        n_dropped_by_zone = 0
        for (label, confidence, bbox_normalized), emb in zip(detections, embeddings):
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
                kept_embeddings.append(emb)
            else:
                n_dropped_by_zone += 1
        detections = kept
        if n_dropped_by_zone:
            logger.info(
                "Exclusion filter dropped %d/%d detection(s) inside zones.",
                n_dropped_by_zone, n_dropped_by_zone + len(kept),
            )

        # --- Norfair tracker (Phase B) ---
        global _tracker, _tracker_params
        tracker_on = bool(tracker_enabled.value)
        if tracker_on:
            reid_on = bool(tracker_reid_enabled.value)
            desired_params = (
                float(tracker_iou_threshold.value),
                int(tracker_max_misses.value),
                max(1, int(tracker_confirm_hits.value)),
                reid_on,
                int(tracker_reid_hit_counter_max.value),
            )
            # Re-instantiate tracker on knob change under lock so the empty-update
            # in process_frame() can't trip on a half-rebuilt tracker.
            with _tracker_lock:
                if _tracker is None or _tracker_params != desired_params:
                    tracker_kwargs = dict(
                        distance_function="iou",
                        distance_threshold=1.0 - desired_params[0],
                        hit_counter_max=desired_params[1],
                        initialization_delay=min(desired_params[2], max(0, desired_params[1] - 1)),
                    )
                    if reid_on:
                        # Keep a short history of embeddings per track and let the
                        # tracker re-match lost tracks by histogram correlation.
                        tracker_kwargs.update(
                            past_detections_length=_REID_PAST_DETECTIONS,
                            reid_distance_function=_reid_embedding_distance,
                            reid_distance_threshold=float(tracker_reid_threshold.value),
                            reid_hit_counter_max=int(tracker_reid_hit_counter_max.value),
                        )
                    _tracker = norfair.Tracker(**tracker_kwargs)
                    _tracker_params = desired_params

                norfair_dets = [
                    norfair.Detection(
                        points=np.array([[xmin, ymin], [xmax, ymax]], dtype=np.float32),
                        scores=np.array([conf, conf], dtype=np.float32),
                        label=label,
                        embedding=emb,
                    )
                    for (label, conf, (ymin, xmin, ymax, xmax)), emb in zip(detections, kept_embeddings)
                ]
                tracked = _tracker.update(detections=norfair_dets)
                matched_ids = {id(d) for d in norfair_dets}

            visible: list[tuple[str, float, list[float], int, int]] = []
            for obj in tracked:
                if obj.id is None:
                    continue
                matched_now = id(obj.last_detection) in matched_ids
                if zones_active and not matched_now:
                    continue
                est = obj.estimate
                x1, x2 = float(est[0][0]), float(est[1][0])
                y1, y2 = float(est[0][1]), float(est[1][1])
                bbox_norm = [min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)]
                if zones_active and _exclusion.bbox_centroid_inside_any(bbox_norm):
                    continue
                label = str(obj.last_detection.label)
                scores = obj.last_detection.scores
                conf = float(np.mean(scores)) if scores is not None else 0.0
                visible.append((label, conf, bbox_norm, int(obj.id), int(obj.age)))

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
                    elif time.time() - _s.splash_armed_at >= _s.splash_delay.value:
                        from .engine import activate_pump  # noqa: PLC0415
                        _s.splash_cooldown_until = time.time() + _s.splash_cooldown.value
                        _s.splash_firing_until = time.time() + _s.splash_duration.value
                        _s.splash_armed_at = 0.0
                        activate_pump(_s.splash_duration.value, _s.pump_duty.value)
                        logger.info("Splash fired for target=%s (duration=%.1fs, cooldown=%.1fs)",
                                    chosen_label, _s.splash_duration.value, _s.splash_cooldown.value)
                        if _event_logger is not None:
                            _event_logger.write({
                                "event": "splash_fired",
                                "label": chosen_label,
                                "duration_seconds": float(_s.splash_duration.value),
                            })
                else:
                    _s.splash_armed_at = 0.0
            else:
                _s.splash_armed_at = 0.0

        dashboard.update(worker_id=worker_pid, inference_time=inference_time)

        # --- Display work ---
        global _last_seen_monotonic_ns, _last_seen_denormalized, latest_ai_detections
        now_mono_ns = time.monotonic_ns()
        ghost_window_ns = max(0, int(ghost_frames_ms.value)) * 1_000_000

        if streaming_active.is_set():
            with cache_lock:
                lores_shape = lowres_frame_cache.pop(timestamp, None)

            detections_denormalized: list[Detection] = []

            _s.tracker_drawables.clear()
            for entry in detections:
                label, confidence, bbox_normalized = entry[0], entry[1], entry[2]
                tid = entry[3] if len(entry) > 3 else None
                age = entry[4] if len(entry) > 4 else None
                if lores_shape is None:
                    break
                if confidence < conf_keep:
                    continue
                if aim_enabled and target_classes and label.strip().lower() not in target_classes:
                    continue
                x, y, w, h = denormalize(bbox_normalized=bbox_normalized, frame_shape=lores_shape)
                if x < 0 or y < 0 or w < 0 or h < 0:
                    logger.warning("Abnormal denormalized bbox: %s → %s", bbox_normalized, (x, y, w, h))
                    continue
                detections_denormalized.append(
                    Detection(label=label, confidence=confidence, bbox=(x, y, w, h)),
                )
                d = norfair.Detection(
                    points=np.array([[x, y], [x + w, y + h]], dtype=np.float32),
                    scores=np.array([confidence], dtype=np.float32),
                    label=label,
                )
                if tid is not None:
                    d.id = tid
                _s.tracker_drawables.append(d)

            if detections_denormalized:
                _last_seen_monotonic_ns = now_mono_ns
                _last_seen_denormalized = detections_denormalized
            elif ghost_window_ns > 0 and (now_mono_ns - _last_seen_monotonic_ns) <= ghost_window_ns:
                detections_denormalized = list(_last_seen_denormalized)

            if ros_node is not None:
                ros_node.publish_detections(detections_denormalized)

            with latest_ai_lock:
                latest_ai_detections = detections_denormalized
        else:
            with cache_lock:
                lowres_frame_cache.pop(timestamp, None)

            if ros_node is not None:
                ros_node.publish_detections([
                    Detection(label=label, confidence=confidence, bbox=list(bbox))
                    for label, confidence, bbox, *_ in detections
                ])

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
        "ghost_frames_ms": int(ghost_frames_ms.value),
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
            with event_clip_queue_lock:
                event_clip_queue.append({
                    "filename": file_path.name,
                    "file": str(file_path.relative_to(django_settings.MEDIA_ROOT)),
                    "frame_count": frame_count,
                    "time": datetime.now().isoformat(),
                    "video_id": video.id,
                })
            logger.info("Event clip saved: %s (%d frames, log=%s)", file_path.name, frame_count,
                        log_path.name if log_path else "no")
        except Exception:
            logger.exception("Failed to save event clip to DB")
    else:
        logger.warning("Event recording produced no output: error=%s", error)


def _submit_yolo(*, nv12_frame: np.ndarray, frame_lores: np.ndarray, rois: list, timestamp: int) -> None:
    if streaming_active.is_set():
        with cache_lock:
            lowres_frame_cache[timestamp] = frame_lores.shape[:2]
    future = inference_pool.submit(run_object_detection, frame_hires=nv12_frame, rois=rois, timestamp=timestamp)
    active_futures.append(future)
    future.add_done_callback(on_done)


def process_frame(*, nv12_frame: np.ndarray, frame_h: int):
    global latest_ai_detections
    global _latest_mask, _latest_mask_shape

    current_time = time.time_ns()

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

    detections_to_show = []

    if os.environ.get("DISABLE_AI"):
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
                    draw_boxes=False,
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

        if app_settings.debug_settings.show_mask or app_settings.debug_settings.show_rois:
            if frame_lores.ndim == 2:
                frame_lores = cv2.merge((frame_lores, frame_lores, frame_lores))
            else:
                frame_lores = frame_lores.copy()

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
                    draw_boxes=False,
                )
    else:
        # No movement — age the tracker with an empty update so a stationary
        # target's track expires consistently while inference is gated off.
        # Guarded by _tracker_lock since on_done() also calls update().
        if tracker_enabled.value:
            with _tracker_lock:
                if _tracker is not None:
                    _tracker.update(detections=[])
        if streaming_active.is_set():
            ghost_window_ns = max(0, int(ghost_frames_ms.value)) * 1_000_000
            now_mono_ns = time.monotonic_ns()
            if ghost_window_ns > 0 and (now_mono_ns - _last_seen_monotonic_ns) <= ghost_window_ns:
                with latest_ai_lock:
                    detections_to_show = list(_last_seen_denormalized)
                    latest_ai_detections = detections_to_show
            else:
                with latest_ai_lock:
                    latest_ai_detections = []
                detections_to_show = []

    if streaming_active.is_set():
        latest_frame.update(frame_lores.copy(), detections_to_show, current_time)


def stream_nonblocking():
    thread_pool.submit(stream_with_ros)


def stream_with_ros():
    try:
        global ros_node

        delay_seconds = 1
        logger.info(f"Starting ROS 2 videostream in {delay_seconds}s...")
        time.sleep(delay_seconds)

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

    logger.info("[DJANGO SHUTDOWN] Processes stopped..")
