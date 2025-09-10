import logging
import multiprocessing as mp
import os
import platform
import random
import subprocess
from ctypes import c_float
from multiprocessing import Event, Queue

from .interfaces import MultiprocessingDequeue, TuningSettings
from .settings import AppSettings, DebugSettings

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

app_settings = AppSettings(debug_settings=DebugSettings(render_bboxes=True))


has_opencl = cv2.ocl.haveOpenCL()
logger.info(f"OpenCV has OpenCL: {has_opencl}")
if has_opencl:
    cv2.ocl.setUseOpenCL(True)

live_stream_enabled = Event()
worker_ready = Event()

NUM_AI_WORKERS: int = 4
preview_downscale_factor = 1
ai_input_size = 640

settings = TuningSettings()
mask_transparency = mp.Value(c_float, 0.5)
# is_mask_streaming_enabled = Event()
is_object_detection_disabled = Event()

# DJANGO_RELOAD_ISSUED = Event()
# DJANGO_RELOAD_SEMAPHORE = Semaphore(NUM_AI_WORKERS)

output_buffer = MultiprocessingDequeue(queue=Queue(maxsize=10))  # todo: make typed
mask_output_buffer = MultiprocessingDequeue[np.ndarray](queue=Queue(maxsize=10))
prob_threshold = mp.Value(c_float, 0.4)
ffmpeg_output: subprocess.Popen | None = None


def start_ffmpeg_writer(width: int, height: int):
    global ffmpeg_output

    # --- Video Stream Parameters ---
    FRAMERATE = 20
    FFMPEG_RTP_URL = "rtp://127.0.0.1:5004"

    codec = "h264_rkmpp" if platform.machine().lower() == "aarch64" else "h264"

    # The full FFmpeg command as a list of arguments
    # Using a list is safer than a single string
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",  # Note: OpenCV uses BGR format
        "-s",
        f"{width}x{height}",
        "-r",
        str(FRAMERATE),
        "-i",
        "-",  # The input comes from a pipe
        "-c:v",
        codec,  # Your hardware encoder
        "-preset",
        "fast",
        "-b:v",
        "2M",
        "-f",
        "rtp",
        FFMPEG_RTP_URL,
    ]

    # --- Starting the FFmpeg Process ---
    # We open a pipe to stdin and redirect stderr to a pipe to read FFmpeg's logs for debugging
    try:
        print("Starting ffmpeg writer...")
        ffmpeg_output = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        print("FFmpeg process started.")
    except FileNotFoundError:
        print("Error: FFmpeg command not found. Make sure it's in your PATH.")
        # Handle the error appropriately
        exit()


class MotionDetector:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            history=settings.foreground_mask_options.mog2_history.value,
            varThreshold=settings.foreground_mask_options.mog2_var_threshold.value,
        )
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # self.morph_kernel = np.ones((3, 3), np.uint8)
        self.pixelcount_threshold = 500

        self.min_area = 500
        self.min_roi_size = int(ai_input_size / preview_downscale_factor)
        self.max_roi_size = int((ai_input_size + 100) / preview_downscale_factor)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def is_moving(self, frame: np.ndarray):
        # motion_ms = get_measure("Detect motion")
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        # lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        # frame = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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
        # cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel, fg_mask)
        # # Connect nearby regions (cat body parts)
        # cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel, fg_mask)
        is_moving = cv2.countNonZero(fg_mask) >= self.pixelcount_threshold
        # motion_ms()
        return is_moving, fg_mask

    def create_rois(self, *, mask: np.ndarray) -> list:
        """
        Finds blobs with connectedComponents, clusters them with constraints,
        and finalizes them to meet min_size requirements.

        Args:
            mask (np.ndarray): The input binary mask.

        Returns:
            list: A list of the final, fully optimized ROIs.
        """
        # 1. Find all individual blobs in the mask (Fast)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)

        # 2. Filter small blobs and collect their initial bounding boxes
        initial_boxes = []

        # if num_labels > 1:
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                # fullness = area / (w * h)
                initial_boxes.append((x, y, w, h))

        if not initial_boxes:
            return []

        # # 3. Cluster the initial boxes with full constraints (Intelligent)
        enable_clustering = True  # todo: make configurable
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
            # Assumes self._expand_roi_to_min_size is your boundary-aware finalization function
            final_roi = expand_roi_to_min_size(min_roi_size=self.min_roi_size, roi=roi_box, img_shape=mask.shape)
            final_rois.append(final_roi)

        # Optional: Sort final ROIs
        final_rois.sort(key=lambda roi: edge_distance(roi=roi, img_shape=mask.shape))

        return final_rois

    def get_bounding_boxes(
        self,
        foreground_mask: np.ndarray,
    ):
        # measure = get_measure("ROI retrieval")
        # res = self.method1_connected_components(foreground_mask)
        # res = self.method1_with_clustering(foreground_mask)
        res = self.create_rois(mask=foreground_mask)
        res = apply_non_max_suppression(boxes=res)
        # res = self.method2_watershed_segmentation(foreground_mask)
        # res = self.method3_mean_shift_clustering(foreground_mask)
        # res = self.method4_adaptive_threshold_contours(foreground_mask)
        # measure()
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
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

        colored_overlay = np.full(frame.shape, overlay_color_rgb, dtype=np.uint8)  # todo: do this only once
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


motion_detector = MotionDetector()
