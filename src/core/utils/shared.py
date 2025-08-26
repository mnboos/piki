import atexit
import dataclasses
import multiprocessing as mp
import logging
import random
from ctypes import c_float, c_int
from multiprocessing import Event, Queue, Condition, Manager, shared_memory, Lock
from typing import TypeVar, Generic
import inspect
import os

# Those most be set BEFORE importing cv2
# https://docs.opencv.org/4.x/d6/dea/tutorial_env_reference.html#autotoc_md974
os.environ["OPENCV_FFMPEG_DEBUG"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "DEBUG"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;rkmpp"

import cv2
import numpy as np

from .func import (
    edge_distance,
    cluster_with_constraints,
    expand_roi_to_min_size,
    apply_non_max_suppression,
)

logger = logging.getLogger(__name__)

logger.info("Setup shared module...")

T = TypeVar("T")

has_opencl = cv2.ocl.haveOpenCL()
logger.info(f"OpenCV has OpenCL: {has_opencl}")
if has_opencl:
    cv2.ocl.setUseOpenCL(True)

class DoubleBuffer:
    """
    A synchronized, shared-memory double buffer for efficient, non-blocking
    frame passing between a high-speed producer and a slower consumer.
    """
    def __init__(self, name: str, shape: tuple, dtype: np.dtype):
        """
        Initializes the two shared memory buffers.
        Args:
            name: A unique name prefix for the shared memory blocks.
            shape: The numpy shape of the data (e.g., (1080, 1920, 3)).
            dtype: The numpy data type (e.g., np.uint8).
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
        size_in_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

        try:
            self._shm_a = shared_memory.SharedMemory(create=True, size=size_in_bytes, name=f"shm_{name}_a")
            self._shm_b = shared_memory.SharedMemory(create=True, size=size_in_bytes, name=f"shm_{name}_b")
        except FileExistsError:
            logger.warning(f"Shared memory for '{name}' already exists. Unlinking and recreating.")
            shared_memory.SharedMemory(name=f"shm_{name}_a").unlink()
            shared_memory.SharedMemory(name=f"shm_{name}_b").unlink()
            self._shm_a = shared_memory.SharedMemory(create=True, size=size_in_bytes, name=f"shm_{name}_a")
            self._shm_b = shared_memory.SharedMemory(create=True, size=size_in_bytes, name=f"shm_{name}_b")

        self._buffer_a = np.ndarray(shape, dtype=dtype, buffer=self._shm_a.buf)
        self._buffer_b = np.ndarray(shape, dtype=dtype, buffer=self._shm_b.buf)
        self._buffers = [self._buffer_a, self._buffer_b]
        self._lock = Lock()
        self._write_index = 0

    def write(self, frame: np.ndarray):
        """Writes a new frame to the current back buffer."""
        self._buffers[self._write_index][:] = frame

    def read_and_swap(self) -> np.ndarray:
        """Atomically swaps buffers and returns a copy of the new front buffer."""
        with self._lock:
            read_idx = self._write_index
            self._write_index = 1 - read_idx
        return self._buffers[read_idx].copy()

    def close(self):
        """Closes and unlinks the shared memory blocks."""
        logger.info(f"Closing and unlinking double buffer '{self.name}'...")
        self._shm_a.close()
        self._shm_a.unlink()
        self._shm_b.close()
        self._shm_b.unlink()

@dataclasses.dataclass
class ForegroundMaskOptions:
    mog2_history = mp.Value(c_int, 500)
    mog2_var_threshold = mp.Value(c_int, 16)
    denoise_kernelsize = mp.Value(c_int, 7)


class TuningSettings:
    def __init__(self):
        self.foreground_mask_options = ForegroundMaskOptions()

    def update(self):
        pass


live_stream_enabled = Event()
worker_ready = Event()

NUM_AI_WORKERS: int = 1
preview_downscale_factor = 1
ai_input_size = 640

settings = TuningSettings()
mask_transparency = mp.Value(c_float, 0.5)
# is_mask_streaming_enabled = Event()
is_object_detection_disabled = Event()

# DJANGO_RELOAD_ISSUED = Event()
# DJANGO_RELOAD_SEMAPHORE = Semaphore(NUM_AI_WORKERS)


class MultiprocessingDequeue(Generic[T]):
    def __init__(self, queue: "Queue[T]") -> None:
        self.queue = queue
        self.condition = Condition()
        self.event = Event()

    def append(self, item: T):
        with self.condition:
            if self.queue.full():
                self.queue.get()
            self.queue.put(item)
            self.condition.notify_all()
        self.event.set()

    def wait_for_data(self):
        self.event.wait()
        self.event.clear()

    def popleft(self) -> T | None:
        with self.condition:
            if not self.queue.empty():
                return self.queue.get()
        return None

    def popleft_blocking(self) -> T:
        with self.condition:
            self.condition.wait(5)
            return self.queue.get()


output_buffer = MultiprocessingDequeue(queue=Queue(maxsize=10))
mask_output_buffer = MultiprocessingDequeue[np.ndarray](queue=Queue(maxsize=10))
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
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv2.CV_32S
        )

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
            final_roi = expand_roi_to_min_size(
                min_roi_size=self.min_roi_size, roi=roi_box, img_shape=mask.shape
            )
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

        colored_overlay = np.full(
            frame.shape, overlay_color_rgb, dtype=np.uint8
        )  # todo: do this only once
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


motion_detector: MotionDetector


def setup_motion_detector():
    global motion_detector
    motion_detector = MotionDetector()


setup_motion_detector()


manager = Manager()

atexit.register(manager.shutdown)


def is_shared_memory_subclass(cls):
    """Helper to safely check if a type is a subclass of SharedMemoryObject."""
    return inspect.isclass(cls) and issubclass(cls, SharedMemoryObject)


class SharedMemoryObject:
    """
    A base class that acts like a dataclass but uses a shared
    dictionary for its state, supporting nested instances.
    """

    def __init__(self, _dict_proxy=None, **kwargs):
        # Initialize the shared dictionary for this instance
        shared_dict = _dict_proxy if _dict_proxy is not None else manager.dict()
        super().__setattr__("_shared_dict", shared_dict)

        all_fields = self.__class__.__annotations__

        # Handle all provided keyword arguments
        for field_name, provided_value in kwargs.items():
            if field_name not in all_fields:
                raise TypeError(
                    f"__init__() got an unexpected keyword argument '{field_name}'"
                )

            self._initialize_field(field_name, all_fields[field_name], provided_value)

        # Handle any fields that were NOT provided and need default initialization
        for field_name, field_type in all_fields.items():
            if field_name not in kwargs:
                self._initialize_field(field_name, field_type)

    def _initialize_field(self, name, type_hint, value=None):
        """Helper method to initialize a single field."""
        is_nested_type = is_shared_memory_subclass(type_hint)

        if is_nested_type:
            # Case 1: An instance was passed (e.g., debug_settings=DebugSettings())
            if isinstance(value, SharedMemoryObject):
                nested_obj = value
                # Adopt the existing object's shared dictionary
                self._shared_dict[name] = nested_obj._shared_dict
                super().__setattr__(name, nested_obj)

            # Case 2: A dictionary was passed (e.g., debug_settings={'render_bboxes': True})
            elif isinstance(value, dict):
                nested_dict = manager.dict()
                self._shared_dict[name] = nested_dict
                nested_obj = type_hint(_dict_proxy=nested_dict, **value)
                super().__setattr__(name, nested_obj)

            # Case 3: Nothing was passed, so create a default empty instance
            elif value is None:
                nested_dict = manager.dict()
                self._shared_dict[name] = nested_dict
                nested_obj = type_hint(_dict_proxy=nested_dict)
                super().__setattr__(name, nested_obj)
            else:
                raise TypeError(
                    f"Argument '{name}' must be a dict or a {type_hint.__name__} instance, not {type(value).__name__}"
                )
        else:
            # It's a primitive type, just assign the value (or None if not provided)
            self._shared_dict[name] = value

    def __getattr__(self, name):
        """Called when accessing an attribute that isn't found normally."""
        # We need to handle the case where we are accessing a nested object instance
        if name in self.__class__.__annotations__ and is_shared_memory_subclass(
            self.__class__.__annotations__[name]
        ):
            return super().__getattribute__(name)

        if name in self._shared_dict:
            return self._shared_dict[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        """Called when setting any attribute."""
        if name not in self.__class__.__annotations__:
            raise AttributeError(
                f"Cannot set new attribute '{name}'. Only defined fields can be set."
            )

        field_type = self.__class__.__annotations__.get(name)
        if is_shared_memory_subclass(field_type):
            raise AttributeError(
                f"Cannot replace nested SharedMemoryObject '{name}'. Modify its attributes instead."
            )

        self._shared_dict[name] = value

    def to_dict(self):
        """Recursively converts the shared object to a regular dictionary."""
        plain_dict = {}
        for key in self.__class__.__annotations__:
            value = getattr(self, key)
            if isinstance(value, SharedMemoryObject):
                plain_dict[key] = value.to_dict()
            else:
                plain_dict[key] = value
        return plain_dict

    def __repr__(self):
        """Provides a nice, readable representation of the object's current state."""
        return f"{self.__class__.__name__}({self.to_dict()})"


class DebugSettings(SharedMemoryObject):
    """The in-memory version of our settings."""

    debug_enabled: bool
    render_bboxes: bool

    def __init__(self, *, render_bboxes: bool, _dict_proxy=None):
        super().__init__(render_bboxes=render_bboxes, _dict_proxy=_dict_proxy)


class AppSettings(SharedMemoryObject):
    debug_settings: DebugSettings

    def __init__(self, *, debug_settings: DebugSettings, _dict_proxy=None):
        super().__init__(debug_settings=debug_settings, _dict_proxy=_dict_proxy)


app_settings = AppSettings(debug_settings=DebugSettings(render_bboxes=True))

