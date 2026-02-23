import atexit
import io
import logging
import multiprocessing as mp
import os
import platform
import signal
import subprocess
import threading
import time
import traceback
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import Lock, shared_memory
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import IO, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import rclpy
    from rclpy.node import Node

import numpy as np

from .func import (
    slice_roi_into_tiles,
)
from .interfaces import Box, DoubleBuffer
from .metrics import LiveMetricsDashboard
from .shared import (
    NUM_AI_WORKERS,
    Detection,
    InferenceOutput,
    OutputResult,
    ai_input_size,
    app_settings,
    cv2,
    is_object_detection_disabled,
    mask_transparency,
    motion_detector,
    output_buffer,
    preview_downscale_factor,
)

logger = logging.getLogger(__name__)

SHM_NAME = "psm_frame_buffer"  # A unique name for our shared memory block
shm_lock = Lock()  # To synchronize access to the shared memory
shared_mem: SharedMemory | None = None  # Will hold the SharedMemory instance
# noinspection PyTypeHints
shared_array: np.typing.NDArray | None = None  # The numpy array view of the shared memory

last_known_bbox = None
last_known_velocity = None
untracked_frames_count = 0
total_untracked_frames_count = 0
velocity_buffer = deque(maxlen=15)

double_buffer: DoubleBuffer | None = None
worker_semaphore = Semaphore(NUM_AI_WORKERS)
ffmpeg_process: subprocess.Popen | None = None
lowres_frame_cache = {}
cache_lock = Lock()


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


def setup_shared_memory_like(frame: np.typing.NDArray):
    global shared_mem
    global shared_array

    """Creates the shared memory block based on the first frame's properties."""
    try:
        # Create a new shared memory block
        size = int(np.prod(frame.shape) * np.dtype(frame.dtype).itemsize)
        shared_mem = SharedMemory(create=True, size=size, name=SHM_NAME)
        logger.info(f"Created shared memory block '{SHM_NAME}' with size {size / 1024 ** 2:.2f} MB")
    except FileExistsError:
        # If it already exists from a previous crashed run, unlink it and retry
        logger.info("Shared memory block already exists, unlinking and recreating.")
        SharedMemory(name=SHM_NAME).unlink()
        size = int(np.prod(frame.shape) * np.dtype(frame.dtype).itemsize)
        shared_mem = SharedMemory(create=True, size=size, name=SHM_NAME)

    # Create a NumPy array that uses the shared memory buffer
    shared_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shared_mem.buf)
    return shared_array


def cleanup_shared_memory():
    """Closes and unlinks the shared memory block on application exit."""
    logger.info("Cleaning up shared memory...")
    if shared_mem:
        shared_mem.close()
        shared_mem.unlink()  # Free the memory block


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

    try:
        frame_h, frame_w = frame_hires.shape[:2]

        from .ai import detect_objects, MODEL_INPUT_TYPE  # noqa: PLC0415

        tiles = slice_roi_into_tiles(
            frame=frame_hires,
            rois=rois,
            tile_size=ai_input_size,
            preview_downscale_factor=preview_downscale_factor,
            model_input_type=MODEL_INPUT_TYPE,
        )
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


def denormalize(bbox_normalized: Sequence[int], frame_shape: Sequence[int]) -> Box:
    frame_height, frame_width, _ = frame_shape
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
    global max_output_timestamp
    global tracker
    active_futures.remove(future)
    try:
        worker_pid, timestamp, inference_time, detections = future.result()

        with cache_lock:
            frame_lores = lowres_frame_cache.pop(timestamp, None)

        if timestamp < max_output_timestamp:
            logger.info(f"Inference result arrived late, discarding (timestamp={timestamp})")
        else:
            max_output_timestamp = timestamp
            detections_denormalized: list[Detection] = []

            for label, confidence, bbox_normalized in detections:
                x, y, w, h = denormalize(bbox_normalized, frame_lores.shape)
                if x < 0 or y < 0 or w < 0 or h < 0:
                    logger.warning("Abnormal denormalized bbox: %s → %s", bbox_normalized, (x, y, w, h))
                    continue

                detections_denormalized.append(
                    Detection(label=label, confidence=confidence, bbox=(x, y, w, h)),
                )

                with tracker_lock:
                    disable_tracking = True
                    if not disable_tracking and label in ["person"] and not tracking.is_set():
                        if tracker is None:
                            tracker_class = cv2.TrackerKCF
                            params = tracker_class.Params()
                            tracker = tracker_class.create(params)
                        tracking.set()
                        tracker.init(frame_lores, (x, y, w, h))

            dashboard.update(worker_id=worker_pid, inference_time=inference_time)

            if ros_node is not None:
                ros_node.publish_detections(detections_denormalized)

            output_buffer.append(
                OutputResult(
                    worker_pid=worker_pid,
                    timestamp=timestamp,
                    frame_lores=frame_lores,
                    detections_denormalized=detections_denormalized,
                ),
            )
    except KeyboardInterrupt:
        logger.info("Shutting down on KeyboardInterrupt in on_done.")
    except:
        traceback.print_exc()
        raise


def process_frame(frame_hires: np.ndarray):  # noqa: C901, PLR0912, PLR0915
    global tracker
    global untracked_frames_count
    global total_untracked_frames_count
    global last_known_bbox

    current_time = time.time_ns()

    # Downscale for motion detection only — MOG2 does not need full resolution.
    # preview_downscale_factor=3 gives 640x360 from 1920x1080.
    frame_lores = cv2.resize(
        frame_hires,
        None,
        fx=1 / preview_downscale_factor,
        fy=1 / preview_downscale_factor,
        interpolation=cv2.INTER_NEAREST,
    )

    if frame_lores is not None and frame_lores.size:
        has_movement, mask = motion_detector.is_moving(frame_lores)

        with tracker_lock:
            is_tracking = tracking.is_set()

        if is_tracking:
            is_coasting = untracked_frames_count
            if not is_coasting:
                assert tracker
                # measure_tracking = get_measure("tracking")
                found, bbox = tracker.update(frame_lores)
                # tracking_duration = measure_tracking(log=False)
                # logger.info(f"Tracked in {tracking_duration} ms: ", found, bbox)

                if found:
                    if not last_known_bbox:
                        last_known_bbox = current_time, (0, 0, 0, 0)

                    last_time, last_bbox = last_known_bbox
                    last_known_bbox = current_time, bbox

                    x, y, w, h = bbox
                    current_pos = np.array([x + w / 2, y + h / 2], dtype=np.float32)
                    x, y, w, h = last_bbox
                    last_pos = np.array([x + w / 2, y + h / 2], dtype=np.float32)

                    dt = current_time - last_time

                    if dt:
                        instantaneous_velocity_vector = (current_pos - last_pos) / dt
                        velocity_buffer.append(instantaneous_velocity_vector)

                    if len(velocity_buffer) > 0:
                        average_velocity_vector = np.mean(list(velocity_buffer), axis=0)

                        # 4. Calculate speed (magnitude) from the AVERAGE vector
                        smoothed_pixels_per_second = np.linalg.norm(average_velocity_vector)

                        stationary_threshold_pixels_per_sec = 5
                        # 5. Apply the stationary threshold
                        if smoothed_pixels_per_second < stationary_threshold_pixels_per_sec:
                            logger.info(
                                "SNAPPING SPEED TO ZEEEEEEEEEEEEEEEROOOOOOOOOO: %d",
                                smoothed_pixels_per_second,
                            )
                            final_speed_pixels_per_sec = 0.0  # Snap to zero if it's just jitter
                        else:
                            final_speed_pixels_per_sec = smoothed_pixels_per_second

                        logger.info(f"Smoothed Speed: {final_speed_pixels_per_sec:.2f} px/s")

                    last_known_bbox = (current_time, bbox)

                    if ros_node is not None:
                        ros_node.publish_detections([Detection(label="tracker", confidence=1, bbox=bbox)])

                    output_buffer.append(
                        OutputResult(
                            worker_pid=0,
                            timestamp=time.monotonic_ns(),
                            frame_lores=frame_lores,
                            detections_denormalized=[Detection(label="tracker", confidence=1, bbox=bbox)],
                        ),
                    )
                else:
                    untracked_frames_count = 1
                    total_untracked_frames_count += 1
            else:
                total_untracked_frames_count += 1
                untracked_frames_count += 1

                total_untracked_frames_threshold = 90
                if total_untracked_frames_count >= total_untracked_frames_threshold:
                    total_untracked_frames_count = 0
                    untracked_frames_count = 0
                    tracker = None
                    tracking.clear()

                untracked_frames_threshold = 30
                if untracked_frames_count > untracked_frames_threshold:
                    untracked_frames_count = 0

                output_buffer.append(
                    OutputResult(
                        worker_pid=0,
                        timestamp=time.monotonic_ns(),
                        frame_lores=frame_lores,
                        detections_denormalized=[],
                    ),
                )

        elif app_settings.debug_settings.debug_enabled or os.environ.get("DISABLE_AI"):
            grayscale_output = True
            if grayscale_output:
                gray = cv2.cvtColor(frame_lores, cv2.COLOR_BGR2GRAY)
                frame_lores = cv2.merge((gray, gray, gray))
            else:
                frame_lores = cv2.cvtColor(frame_lores, cv2.COLOR_BGR2RGB)

            buf_highlighted = motion_detector.highlight_movement_on(
                frame=frame_lores,
                mask=mask,
                overlay_color_rgb=(
                    147,
                    20,
                    255,
                ),
                transparency_factor=mask_transparency.value,
                draw_boxes=True,
            )
            mode = "stream"
            if mode == "mask":
                output_buffer.append(
                    OutputResult(
                        worker_pid=0,
                        timestamp=current_time,
                        frame_lores=buf_highlighted,
                        detections_denormalized=[],
                    ),
                )
            else:
                output_buffer.append(
                    OutputResult(
                        worker_pid=0,
                        timestamp=current_time,
                        frame_lores=frame_lores,
                        detections_denormalized=[],
                    ),
                )
        elif has_movement:
            try:
                timestamp = time.monotonic_ns()

                rois = motion_detector.create_rois(mask=mask)
                if rois:
                    # Skip if inference is already busy — drop the frame rather than queue up.
                    # The BPU will finish the current batch faster than we can accumulate frames.
                    if not active_futures:
                        with cache_lock:
                            lowres_frame_cache[timestamp] = frame_lores

                        # Pass frame_hires directly — no shared memory, no pickle.
                        future = inference_pool.submit(
                            run_object_detection,
                            frame_hires=frame_hires.copy(),
                            rois=rois,
                            timestamp=timestamp,
                        )
                        active_futures.append(future)
                        future.add_done_callback(on_done)
            except:
                traceback.print_exc()
                raise
        elif not has_movement and not active_futures:
            output_buffer.append(
                OutputResult(worker_pid=0, timestamp=0, frame_lores=frame_lores, detections_denormalized=[]),
            )


class StreamingOutput(io.BufferedIOBase):
    def write(self, buf_hires: bytes) -> None:
        if self.closed:
            raise RuntimeError("Stream is closed!! ")

        if len(active_futures) == NUM_AI_WORKERS:
            return

        decoded_frame = cv2.imdecode(np.frombuffer(buf_hires, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded_frame == True:  # noqa: E712
            process_frame(frame_hires=decoded_frame)


input_buffer = StreamingOutput()


def stream_nonblocking():
    thread_pool.submit(stream_with_ros)


def stream_with_ros():
    try:
        global ros_node
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        import json
        from cv_bridge import CvBridge

        delay_seconds = 5
        logger.info(f"Starting ROS 2 videostream in {delay_seconds}s...")
        time.sleep(delay_seconds)

        print("------------!!!!!!!!!!!!! STREAM (ROS 2)")
        high_res_w, high_res_h = 640, 480

        class PikiVisionNode(Node):
            def __init__(self):
                super().__init__('piki_vision_node')
                # Default to a pre-resized hardware ISP stream to save CPU/GPU overhead
                topic_name = os.environ.get('ROS_IMAGE_TOPIC', '/camera/left/image_raw_640x480')
                self.bridge = CvBridge()

                try:
                    from hbm_img_msgs.msg import HbmMsg1080P
                    self.get_logger().info(f"Using zero-copy HbmMsg1080P for topic: {topic_name}")
                    self.subscription = self.create_subscription(
                        HbmMsg1080P,
                        topic_name,
                        self.listener_callback_hbm,
                        10
                    )
                except ImportError:
                    self.get_logger().info(
                        f"hbm_img_msgs not found, falling back to sensor_msgs.msg.Image for topic: {topic_name}")
                    self.subscription = self.create_subscription(
                        Image,
                        topic_name,
                        self.listener_callback,
                        10
                    )

                # Publisher for tracking target (for servos)
                self.target_pub = self.create_publisher(String, '/piki/target_detections', 10)

            def listener_callback_hbm(self, msg):
                try:
                    # hbm_img_msgs usually contains NV12 image data in its 'data' field.
                    # Convert NV12 to BGR for process_frame.
                    # If your Hobot YOLO model expects NV12 natively, you can bypass this cvtColor
                    # entirely and pass nv12_data straight to process_frame/detect_objects.
                    nv12_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height * 3 // 2, msg.width))

                    # NOTE: Hardware VPS should ideally handle this NV12->BGR and resize.
                    # As a fallback, CPU cvtColor is used.
                    cv_image = cv2.cvtColor(nv12_data, cv2.COLOR_YUV2BGR_NV12)

                    if msg.width == high_res_w and msg.height == high_res_h:
                        # Skip GPU resize since the ISP already resized it for us!
                        frame_hires = cv_image
                    else:
                        # Offload image processing to Mali GPU via OpenCL (T-API) if ISP resize wasn't used
                        umat_image = cv2.UMat(cv_image)
                        umat_resized = cv2.resize(umat_image, (high_res_w, high_res_h))
                        frame_hires = umat_resized.get()

                    process_frame(frame_hires)
                except Exception as e:
                    self.get_logger().error(f'Error processing zero-copy HBM image: {e}')
                    traceback.print_exc()

            def listener_callback(self, msg):
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                    if cv_image.shape[1] == high_res_w and cv_image.shape[0] == high_res_h:
                        # Skip GPU resize since the ISP already gave us the correct size
                        frame_hires = cv_image
                    else:
                        # Offload image processing to Mali GPU via OpenCL (T-API)
                        umat_image = cv2.UMat(cv_image)

                        # Example of where stereo rectification (cv2.remap) would go.
                        # It will run on the GPU automatically.
                        # umat_rectified = cv2.remap(umat_image, map1, map2, cv2.INTER_LINEAR)

                        umat_resized = cv2.resize(umat_image, (high_res_w, high_res_h))
                        frame_hires = umat_resized.get()

                    process_frame(frame_hires)
                except Exception as e:
                    self.get_logger().error(f'Error processing image: {e}')
                    traceback.print_exc()

            def publish_detections(self, detections):
                # Serialize detections to JSON and publish
                data = [{"label": d.label, "confidence": float(d.confidence), "bbox": d.bbox} for d in detections]
                msg = String()
                msg.data = json.dumps(data)
                self.target_pub.publish(msg)

        rclpy.init(args=None)
        ros_node = PikiVisionNode()
        rclpy.spin(ros_node)
    except Exception as e:
        logger.exception(f"ROS 2 streaming failed: {e}")
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


@atexit.register
def cleanup():
    logger.info("[DJANGO SHUTDOWN] Stopping processes.....")

    input_buffer.close()

    thread_pool.shutdown(wait=True, cancel_futures=True)
    inference_pool.shutdown(wait=True, cancel_futures=True)
    if ffmpeg_process:
        ffmpeg_process.kill()
    cleanup_shared_memory()

    logger.info("[DJANGO SHUTDOWN] Processes stopped..")
