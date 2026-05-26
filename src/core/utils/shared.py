import logging
import multiprocessing as mp
import os
import threading
import time
from collections import deque
from collections.abc import Sequence
from ctypes import c_float
from multiprocessing import Event
from typing import NamedTuple

from .interfaces import TuningSettings
from .settings import AppSettings, AimSettings, DebugSettings

# Those most be set BEFORE importing cv2
# https://docs.opencv.org/4.x/d6/dea/tutorial_env_reference.html#autotoc_md974
os.environ["OPENCV_FFMPEG_DEBUG"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "DEBUG"

import cv2
import numpy as np

from .func import (
    cluster_with_constraints,
    edge_distance,
    expand_roi_to_min_size,
)

logger = logging.getLogger(__name__)
logger.info("Setup shared module...")

app_settings = AppSettings(
    debug_settings=DebugSettings(show_boxes=True),
    aim_settings=AimSettings(target_classes=[], servo_enabled=False, target_lock_duration=3.0,
                             aim_confidence=0.4, pan_invert=False, tilt_invert=False),
)


has_opencl = cv2.ocl.haveOpenCL()
logger.info(f"OpenCV has OpenCL: {has_opencl}")
if has_opencl:
    cv2.ocl.setUseOpenCL(True)

worker_ready = Event()

# BPU is a single shared hardware resource — multiple competing processes hurt
# throughput more than they help. One inference thread feeding tiles sequentially
# is faster because the BPU pipeline is already internally pipelined.
NUM_AI_WORKERS: int = 1

# Run MOG2 motion detection on 1/3 scale (640x360) instead of full 1920x1080.
# Motion detection doesn't need full resolution — this saves significant CPU.
# The hi-res frame is still stored in shared memory for full-res tile slicing.
preview_downscale_factor = 2

ai_input_size = 640

settings = TuningSettings()
servo_pan = mp.Value(c_float, 0.0)   # current pan angle in degrees
servo_tilt = mp.Value(c_float, 0.0)  # current tilt angle in degrees
servo_kalman_pan = mp.Value(c_float, 0.0)   # Kalman predicted pan (lookahead target)
servo_kalman_tilt = mp.Value(c_float, 0.0)  # Kalman predicted tilt (lookahead target)
servo_pid_kp = mp.Value(c_float, 1.0)  # positional gain (1.0 = snap directly to target)
servo_pid_ki = mp.Value(c_float, 0.0)  # integral gain
servo_pid_kd = mp.Value(c_float, 0.0)  # derivative gain (raise to reduce jitter)
servo_dead_zone = mp.Value(c_float, 1.5)      # degrees: changes smaller than this in both axes are ignored
servo_kalman_process_noise = mp.Value(c_float, 10.0)   # deg/s² — how quickly velocity may change (lower = less jitter)
servo_kalman_meas_noise = mp.Value(c_float, 5.0)       # deg   — position measurement uncertainty (higher = smoother)
servo_kalman_lookahead_ms = mp.Value(c_float, 50.0)    # ms    — servo lag to compensate for (0 = off)
vertical_angle_offset = mp.Value(c_float, 0.0)          # deg   — tilt offset to compensate for mounting height
servo_aim_confidence = mp.Value(c_float, 0.4)            # minimum confidence to lock onto a target
# Servo direction inversion, mirrored from app_settings.aim_settings into shared
# memory so the 60Hz servo loop reads them without a SyncManager IPC round-trip
# (mp.Value has no bool type, so 0/1 ints — same pattern as tracker_enabled).
servo_pan_invert = mp.Value("i", 0)
servo_tilt_invert = mp.Value("i", 0)
# is_mask_streaming_enabled = Event()
is_object_detection_disabled = Event()

# Set while at least one viewer is connected (WebRTC subscriber or replay).
# webrtc.py and replay flip it; nothing in the inference / motion pipeline
# reads it post-WebRTC, but other modules still gate on it.
streaming_active = threading.Event()


# Set while at least one WebRTC peer is connected. Gates the hardware H.264
# encoder in `process_frame()` so we don't burn VPU cycles when nobody's
# watching. webrtc.py keeps this in lock-step with streaming_active.
webrtc_active = threading.Event()

# Set by the passthrough encoder when the browser requests a keyframe (PLI/FIR).
# The ROS callback checks this before encoding and forces an IDR on the VPU.
webrtc_keyframe_requested = threading.Event()

# Target frame rate for the WebRTC video stream. The hardware encoder always
# sees frames at the camera's native rate (~30 fps); when this is lower we
# skip encode calls in process_frame() so the wire bitrate scales linearly
# with frame rate. Set to 30 (or higher) to disable skipping.
webrtc_target_fps = mp.Value("i", 30)

# Set while recording pipeline frames to a video file.
recording_active = threading.Event()
# Set while replaying a video through the pipeline (replay drives the WebRTC encoder).
replaying_active = threading.Event()


class FPSCounter:
    """Rolling-window FPS counter (thread-safe)."""

    def __init__(self, window: float = 2.0):
        self._ts: deque[float] = deque()
        self._window = window
        self._lock = threading.Lock()

    def tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._ts.append(now)
            cutoff = now - self._window
            while self._ts and self._ts[0] < cutoff:
                self._ts.popleft()

    @property
    def fps(self) -> float:
        with self._lock:
            n = len(self._ts)
            if n < 2:
                return 0.0
            return (n - 1) / (self._ts[-1] - self._ts[0])


fps_counter = FPSCounter()

# DJANGO_RELOAD_ISSUED = Event()
# DJANGO_RELOAD_SEMAPHORE = Semaphore(NUM_AI_WORKERS)


class Detection(NamedTuple):
    label: str
    confidence: float
    bbox: Sequence[float]
    mask_centroid: "tuple[float, float] | None" = None
    mask_polygon: "list[list[float]] | None" = None  # [[x1,y1,...], ...] contours, normalized


class InferenceOutput(NamedTuple):
    worker_pid: int
    timestamp: int
    avg_duration: int
    detections: list[Detection]


# Detection confidence thresholds with hysteresis:
#   prob_threshold      = "enter" threshold (default 0.40) — required to start
#                          a lock, count toward the min-streak, or trigger an
#                          event recording.
#   prob_threshold_keep = "keep"  threshold (default 0.25) — a confirmed lock
#                          survives while matched detections stay at or above
#                          this value.  Also the floor used by the YOLO
#                          post-processor so low-conf candidates reach
#                          on_done() for hysteresis to evaluate.
prob_threshold = mp.Value(c_float, 0.4)
prob_threshold_keep = mp.Value(c_float, 0.25)

# Per-label same-class consecutive-hit counter requirement (gating noise).
min_consecutive_frames = mp.Value("i", 2)

# Exponential-moving-average factor for smoothing the locked-target bbox.
bbox_ema_alpha = mp.Value(c_float, 0.7)

# EMA factor for smoothing raw detection center (x, y) per track before
# publishing to the frontend.  Formula: smoothed = α·new + (1-α)·old.
# 1.0 = raw pass-through; lower = smoother / laggier.
coord_ema_alpha = mp.Value(c_float, 1.0)

# --- OC-Sort tracker (Phase B) ---
# When enabled, on_done() routes detections through an OC-Sort tracker whose
# confirmed tracks become the source of identity for the technical log and the
# source of truth for downstream consumers (aim, recording trigger).
tracker_enabled = mp.Value("i", 1)  # 1 = on, 0 = off (mp.Value has no bool type)
tracker_iou_threshold = mp.Value(c_float, 0.3)
tracker_max_misses = mp.Value("i", 30)
tracker_confirm_hits = mp.Value("i", 3)
tracker_delta_t = mp.Value("i", 3)
tracker_inertia = mp.Value(c_float, 0.2)


class MotionDetector:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            history=settings.foreground_mask_options.mog2_history.value,
            varThreshold=settings.foreground_mask_options.mog2_var_threshold.value,
        )
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # self.morph_kernel = np.ones((3, 3), np.uint8)
        self.min_roi_size = int(ai_input_size / preview_downscale_factor)
        self.max_roi_size = int((ai_input_size + 100) / preview_downscale_factor)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def reset(self):
        """Discard the learned background model and start fresh."""
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            history=settings.foreground_mask_options.mog2_history.value,
            varThreshold=settings.foreground_mask_options.mog2_var_threshold.value,
        )

    def is_moving(self, frame: np.ndarray):
        import os, time  # noqa: PLC0415, E401
        _t0 = time.perf_counter() if os.environ.get("PIKI_PROFILE") else None

        gray = frame if frame.ndim == 2 or frame.shape[2] == 1 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = self.clahe.apply(gray)

        denoise_kernelsize = settings.foreground_mask_options.denoise_kernelsize.value
        if denoise_kernelsize >= 1:
            kernel = (denoise_kernelsize,) * 2
            cv2.GaussianBlur(frame, kernel, 0, frame)

        fg_mask = self.backSub.apply(frame)
        # Remove noise
        cv2.dilate(fg_mask, self.morph_kernel, iterations=1, dst=fg_mask)
        cv2.erode(fg_mask, self.morph_kernel, iterations=2, dst=fg_mask)
        cv2.dilate(fg_mask, self.morph_kernel, iterations=1, dst=fg_mask)
        is_moving = cv2.countNonZero(fg_mask) >= settings.foreground_mask_options.pixelcount_threshold.value

        if _t0 is not None:
            logger.info("PERF stage=motion_detect ms=%.2f", (time.perf_counter() - _t0) * 1000)
        return is_moving, fg_mask

    def create_rois(self, *, mask: np.ndarray) -> list:
        """Find, cluster and finalilze blobs with connectedComponents.

        Args:
            mask (np.ndarray): The input binary mask.

        Returns:
            list: A list of the final, fully optimized ROIs.

        """
        import os, time  # noqa: PLC0415, E401
        _t0 = time.perf_counter() if os.environ.get("PIKI_PROFILE") else None

        # 1. Find all individual blobs in the mask (Fast)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)

        # 2. Filter small blobs and collect their initial bounding boxes
        initial_boxes = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= settings.foreground_mask_options.min_area.value:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                initial_boxes.append((x, y, w, h))

        if not initial_boxes:
            return []

        # 3. Cluster the initial boxes with full constraints (Intelligent)
        enable_clustering = True  # TODO(mnboos): make configurable
        if enable_clustering:
            clustered_rois = cluster_with_constraints(
                boxes=initial_boxes,
                max_dimension=self.max_roi_size,
            )
        else:
            clustered_rois = initial_boxes

        # 4. Finalize ROIs to enforce minimum size and handle edge cases (Format for AI)
        final_rois = []
        for roi_box in clustered_rois:
            final_roi = expand_roi_to_min_size(min_roi_size=self.min_roi_size, roi=roi_box, img_shape=mask.shape)
            final_rois.append(final_roi)

        # Optional: Sort final ROIs
        final_rois.sort(key=lambda roi: edge_distance(roi=roi, img_shape=mask.shape))

        if _t0 is not None:
            logger.info("PERF stage=roi_create ms=%.2f", (time.perf_counter() - _t0) * 1000)
        return final_rois


# --- Event-triggered recording state ---
event_recording_enabled = threading.Event()
event_recording_active = threading.Event()
event_recording_cooldown_until = 0.0

event_pre_buffer_seconds = mp.Value(c_float, 5.0)
event_post_trigger_seconds = mp.Value(c_float, 10.0)
event_cooldown_seconds = mp.Value(c_float, 30.0)

event_trigger_classes: list[str] = []
event_trigger_classes_lock = threading.Lock()

event_clip_queue: deque[dict] = deque(maxlen=20)
event_clip_queue_lock = threading.Lock()

# --- Splash relay state ---
splash_enabled = threading.Event()
splash_cooldown_until = 0.0
splash_armed_at = 0.0
splash_firing_until = 0.0
splash_delay = mp.Value(c_float, 0.5)
splash_duration = mp.Value(c_float, 1.0)
splash_cooldown = mp.Value(c_float, 10.0)
pump_duty = mp.Value(c_float, 100.0)               # PWM duty cycle for pump (0–100%)

motion_detector = MotionDetector()
