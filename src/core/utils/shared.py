import logging
import multiprocessing as mp
import os
import random
import threading
import time
from collections import deque
from collections.abc import Sequence
from ctypes import c_float
from multiprocessing import Event
from typing import NamedTuple, Optional

from .interfaces import TuningSettings
from .settings import AppSettings, AimSettings, DebugSettings

# Those most be set BEFORE importing cv2
# https://docs.opencv.org/4.x/d6/dea/tutorial_env_reference.html#autotoc_md974
os.environ["OPENCV_FFMPEG_DEBUG"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "DEBUG"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;rkmpp"

import cv2
import numpy as np

from .func import (
    apply_non_max_suppression,
    cluster_with_constraints,
    edge_distance,
    expand_roi_to_min_size,
)

logger = logging.getLogger(__name__)
logger.info("Setup shared module...")

app_settings = AppSettings(
    debug_settings=DebugSettings(show_boxes=True, show_mask=False, show_rois=False),
    aim_settings=AimSettings(target_classes=[], servo_enabled=False, target_lock_duration=3.0),
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
mask_transparency = mp.Value(c_float, 0.5)
servo_pan = mp.Value(c_float, 0.0)   # current pan angle in degrees
servo_tilt = mp.Value(c_float, 0.0)  # current tilt angle in degrees
servo_kalman_pan = mp.Value(c_float, 0.0)   # Kalman predicted pan (lookahead target)
servo_kalman_tilt = mp.Value(c_float, 0.0)  # Kalman predicted tilt (lookahead target)
servo_pid_kp = mp.Value(c_float, 1.0)  # proportional gain (1.0 = instant, like previous default)
servo_pid_ki = mp.Value(c_float, 0.0)  # integral gain
servo_pid_kd = mp.Value(c_float, 0.0)  # derivative gain (raise to reduce jitter)
servo_dead_zone = mp.Value(c_float, 1.5)      # degrees: changes smaller than this in both axes are ignored
servo_kalman_process_noise = mp.Value(c_float, 10.0)   # deg/s² — how quickly velocity may change
servo_kalman_meas_noise = mp.Value(c_float, 5.0)       # deg   — position measurement uncertainty
servo_kalman_lookahead_ms = mp.Value(c_float, 50.0)    # ms    — servo lag to compensate for (0 = off)
# is_mask_streaming_enabled = Event()
is_object_detection_disabled = Event()

# Set while at least one MJPEG client is connected.  When clear, all
# display-only work (frame caching, bbox rendering, latest_frame updates)
# is skipped so the inference/motion-detection loop runs at full speed.
streaming_active = threading.Event()

# Set while recording pipeline frames to a video file.
recording_active = threading.Event()
# Set while replaying a video through the pipeline (replay thread owns latest_frame).
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
    # ("tracker", 1, bbox)
    label: str
    confidence: float
    bbox: Sequence[int]


class InferenceOutput(NamedTuple):
    worker_pid: int
    timestamp: int
    avg_duration: int
    detections: list[Detection]


class LatestFrame:
    def __init__(self):
        self.frame: Optional[np.ndarray] = None
        self.detections: list[Detection] = []
        self.timestamp: int = 0
        self.condition = threading.Condition()

    def update(self, frame: np.ndarray, detections: list[Detection], timestamp: int):
        with self.condition:
            self.frame = frame
            self.detections = detections
            self.timestamp = timestamp
            self.condition.notify_all()

    def get(self):
        with self.condition:
            return self.frame, self.detections, self.timestamp

    def wait_for_frame(self, last_timestamp: int):
        with self.condition:
            self.condition.wait_for(lambda: self.timestamp > last_timestamp, timeout=1.0)
            return self.frame, self.detections, self.timestamp


latest_frame = LatestFrame()
latest_debug_frame = LatestFrame()  # raw/distorted frame for the debug video feed
prob_threshold = mp.Value(c_float, 0.4)


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

    def get_bounding_boxes(
        self,
        foreground_mask: np.ndarray,
    ):
        res = self.create_rois(mask=foreground_mask)
        res = apply_non_max_suppression(boxes=res)
        return res

    def highlight_movement_on(
        self,
        *,
        frame: np.ndarray,
        mask: np.ndarray,
        transparency_factor: float = 0.4,
        overlay_color_rgb: tuple[int, int, int] = (255, 0, 0),
        draw_boxes: bool = True,
    ) -> np.ndarray:
        if draw_boxes:
            boxes = self.get_bounding_boxes(mask)
            for x, y, w, h in boxes:
                # rect_color = (0, 0, 255)
                rect_color = (
                    random.randint(0, 255),  # noqa: S311
                    random.randint(0, 255),  # noqa: S311
                    random.randint(0, 255),  # noqa: S311
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

        colored_overlay = np.full(frame.shape, overlay_color_rgb, dtype=np.uint8)  # TODO(mnboos): do this only once
        blended = cv2.addWeighted(
            frame,
            transparency_factor,
            colored_overlay,
            1 - transparency_factor,
            0,
        )
        return np.where(
            mask[:, :, None] != 0,
            blended,
            frame,
        )


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

motion_detector = MotionDetector()
