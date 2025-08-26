import os
import signal
import threading
from collections import deque, namedtuple
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import shared_memory, Lock, Queue
import subprocess
import multiprocessing as mp
import platform
import io
import time
import atexit
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np
import traceback
import logging

from .func import (
    get_padded_roi_images,
)
from .metrics import LiveMetricsDashboard
from .shared import (
    DoubleBuffer,
    mask_transparency,
    app_settings,
    output_buffer,
    mask_output_buffer,
    live_stream_enabled,
    motion_detector,
    NUM_AI_WORKERS,
    preview_downscale_factor,
    ai_input_size,
    is_object_detection_disabled,
    cv2,
)

logger = logging.getLogger(__name__)

SHM_NAME = "psm_frame_buffer"  # A unique name for our shared memory block
shm_lock = Lock()  # To synchronize access to the shared memory
shared_mem: SharedMemory | None = None  # Will hold the SharedMemory instance
shared_array: Optional[np.typing.NDArray] = (
    None  # The numpy array view of the shared memory
)

last_known_bbox = None
last_known_velocity = None
untracked_frames_count = 0
total_untracked_frames_count = 0
velocity_buffer = deque(maxlen=15)


double_buffer: DoubleBuffer | None = None
work_token_queue = Queue(maxsize=NUM_AI_WORKERS)
ffmpeg_process: subprocess.Popen | None = None
lowres_frame_cache = {}
cache_lock = Lock()


def init_worker():
    pid = os.getpid()
    ppid = os.getppid()
    logger.info(f"[Worker-{pid}]: Setup")

    def f():
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()

    @atexit.register
    def _cleanup():
        logger.info(f"[Worker-{pid}] Shutting down....")


tracker_lock = Lock()
tracking = mp.Event()
coasting = mp.Event()
tracker: Optional[cv2.Tracker] = None
process_pool = ProcessPoolExecutor(max_workers=NUM_AI_WORKERS, initializer=init_worker)
thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piki-streamer")
# worker_slot_semaphore = Semaphore(NUM_AI_WORKERS)

def frame_producer(ffmpeg_stdout, buffer_instance: DoubleBuffer):
    """
    Reads raw frames from FFmpeg's stdout pipe and writes them into a DoubleBuffer.
    """
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
        logger.error("Exception in producer thread:")
        traceback.print_exc()
        raise
    logger.info("Producer finished.")


def setup_shared_memory(frame_shape, frame_dtype):
    global shared_mem
    global shared_array

    """Creates the shared memory block based on the first frame's properties."""
    try:
        # Create a new shared memory block
        size = int(np.prod(frame_shape) * np.dtype(frame_dtype).itemsize)
        shared_mem = SharedMemory(create=True, size=size, name=SHM_NAME)
        logger.info(
            f"Created shared memory block '{SHM_NAME}' with size {size / 1024**2:.2f} MB"
        )
    except FileExistsError:
        # If it already exists from a previous crashed run, unlink it and retry
        logger.info("Shared memory block already exists, unlinking and recreating.")
        SharedMemory(name=SHM_NAME).unlink()
        size = int(np.prod(frame_shape) * np.dtype(frame_dtype).itemsize)
        shared_mem = SharedMemory(create=True, size=size, name=SHM_NAME)

    # Create a NumPy array that uses the shared memory buffer
    shared_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shared_mem.buf)
    return shared_array


def cleanup_shared_memory():
    """Closes and unlinks the shared memory block on application exit."""
    logger.info("Cleaning up shared memory...")
    if shared_mem:
        shared_mem.close()
        shared_mem.unlink()  # Free the memory block


active_futures: list[Future] = []
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

    def measure(log: bool = True):
        end = time.perf_counter()
        ms = str(round((end - now) * 1000, 2)).rjust(5)
        if log:
            logger.info(f"{description}: {ms} ms!")
        return ms

    return measure


def run_object_detection(
    shape: tuple,
    dtype: np.dtype,
    rois,
    timestamp: int,
):
    worker_pid = mp.current_process().pid
    if is_object_detection_disabled.set():
        return worker_pid, timestamp, None, (0, [])

    existing_shm = None
    try:
        # --- Shared Memory Access ---
        existing_shm = shared_memory.SharedMemory(name=SHM_NAME)
        frame_in_shm = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        with shm_lock:
            frame_hires = frame_in_shm.copy()

        frame_h, frame_w, _ = frame_hires.shape

        # --- AI Processing ---
        padded_images_and_details = get_padded_roi_images(
            frame=frame_hires,
            rois=rois,
            target_size=ai_input_size,
            preview_downscale_factor=preview_downscale_factor,
        )
        print("images to detect: ", len(padded_images_and_details))

        total_duration = 0
        all_detections = []

        from .ai import detect_objects

        for img, scale, effective_origin in padded_images_and_details[:1]:
            eff_orig_x, eff_orig_y = effective_origin

            duration, detections = detect_objects(img)
            total_duration += duration

            # 'detections' is a list of (label, confidence, local_pixel_bbox)
            # where local_pixel_bbox is (x, y, w, h)
            for label, confidence, local_pixel_bbox in detections:
                # --- START: Correct Transformation Logic ---

                # 1. Unpack the LOCAL PIXEL coordinates (x, y, w, h) from ai.py
                local_px_xmin, local_px_ymin, local_px_xmax, local_px_ymax = (
                    local_pixel_bbox
                )

                # 3. Apply the 'scale' factor from the resizing step
                scaled_px_xmin = local_px_xmin * scale
                scaled_px_ymin = local_px_ymin * scale
                scaled_px_xmax = local_px_xmax * scale
                scaled_px_ymax = local_px_ymax * scale

                # 4. Translate by the 'effective_origin' to get GLOBAL PIXEL coordinates
                global_px_xmin = scaled_px_xmin + eff_orig_x
                global_px_ymin = scaled_px_ymin + eff_orig_y
                global_px_xmax = scaled_px_xmax + eff_orig_x
                global_px_ymax = scaled_px_ymax + eff_orig_y

                # 5. Re-normalize the FINAL GLOBAL pixel coordinates for views.py
                final_norm_coords = [
                    global_px_ymin / frame_h,
                    global_px_xmin / frame_w,
                    global_px_ymax / frame_h,
                    global_px_xmax / frame_w,
                ]

                all_detections.append((label, confidence, final_norm_coords))
                # --- END: Correct Transformation Logic ---

        avg_duration = (
            0
            if not len(padded_images_and_details)
            else total_duration // len(padded_images_and_details)
        )
        result = avg_duration, all_detections
        frame_lores = cv2.resize(
            frame_hires,
            None,
            fx=1 / preview_downscale_factor,
            fy=1 / preview_downscale_factor,
            interpolation=cv2.INTER_NEAREST,
        )

        #return worker_pid, timestamp, frame_lores, result
        return worker_pid, timestamp, result

    except Exception as e:
        logger.info(
            f"!!!!!!!!!!!!!!! FATAL ERROR IN AI WORKER (PID: {os.getpid()}) !!!!!!!!!!!!!!"
        )
        traceback.print_exc()
        raise e
    finally:
        if existing_shm is not None:
            existing_shm.close()


def denormalize(bbox_normalized, frame_shape):
    frame_height, frame_width, _ = frame_shape
    ymin, xmin, ymax, xmax = bbox_normalized

    # 2. Clamp values to the [0.0, 1.0] range to prevent errors
    ymin = max(0.0, ymin)
    xmin = max(0.0, xmin)
    ymax = min(1.0, ymax)
    xmax = min(1.0, xmax)

    # 3. Denormalize to get PIXEL coordinates for the CURRENT frame
    left = int(xmin * frame_width)
    top = int(ymin * frame_height)
    right = int(xmax * frame_width)
    bottom = int(ymax * frame_height)
    width = right - left
    height = bottom - top
    tp = namedtuple("Box", "x y w h")
    return tp(left, top, width, height)


def denormalize_detections(detections, frame_shape):
    detections_denormalized = []
    for label, confidence, bbox_normalized in detections:
        x, y, w, h = denormalize(bbox_normalized, frame_shape)

        detections_denormalized.append((label, confidence, (x, y, w, h)))
    return detections_denormalized


def on_done(future: Future):
    global max_output_timestamp
    global tracker
    active_futures.remove(future)

    # logger.info(f"HANDLE DONE: {os.getpid()}")

    try:
        #worker_pid, timestamp, frame_lores, detected_objects = future.result()
        worker_pid, timestamp, detected_objects = future.result()

        with cache_lock:
            # .pop() is atomic and perfect for this, returns None if key is missing
            frame_lores = lowres_frame_cache.pop(timestamp, None)

        if timestamp < max_output_timestamp:
            logger.info(f"Worker-{worker_pid} was slow, the results came too late :(")
        else:
             max_output_timestamp = timestamp

             if detected_objects:
                inference_time, detections = detected_objects

                detections_denormalized = []

                frame_height, frame_width, _ = frame_lores.shape
                for label, confidence, bbox_normalized in detections:
                    x, y, w, h = denormalize(bbox_normalized, frame_lores.shape)
                    if x < 0 or y < 0 or w < 0 or h < 0:
                        print(
                            "something has been denormalized abnormally: ",
                            frame_lores.shape,
                            bbox_normalized,
                            (x, y, w, h),
                        )
                        continue

                    detections_denormalized.append((label, confidence, (x, y, w, h)))

                    with tracker_lock:
                        disable_tracking = True
                        if (
                            not disable_tracking
                            and label in ["person"]
                            and not tracking.is_set()
                        ):
                            if tracker is None:
                                tracker_class = cv2.TrackerKCF
                                # tracker_class = cv2.TrackerCSRT
                                params = tracker_class.Params()
                                logger.info(
                                    "TrackerKCF params: %s",
                                    {
                                        k: getattr(params, k)
                                        for k in dir(params)
                                        if not k.startswith("_")
                                    },
                                )
                                logger.info("orig params: %s", dir(params))

                                # params.max_patch_size = 1024**2

                                tracker = tracker_class.create(params)
                            logger.info("Tracker init shape: %s", frame_lores.shape)
                            tracking.set()
                            tracker.init(frame_lores, (x, y, w, h))
                dashboard.update(worker_id=worker_pid, inference_time=inference_time)
                if live_stream_enabled.is_set():
                    output_buffer.append(
                        (worker_pid, timestamp, frame_lores, detections_denormalized)
                    )
    except BrokenProcessPool:
        logger.info("Pool already broken, when future was done. Shutting down...")
        traceback.print_exc()
    except KeyboardInterrupt:
        logger.info("Future done, shutting down....")
    except:
        traceback.print_exc()
        raise
    finally:
        work_token_queue.put(None)


def process_frame(frame_hires: np.ndarray):
    global shared_array
    global tracker
    global untracked_frames_count
    global total_untracked_frames_count
    global last_known_bbox
    global last_known_velocity

    current_time = time.monotonic()

    with shm_lock:
        if shared_array is None:
            shared_array = setup_shared_memory(frame_hires.shape, frame_hires.dtype)

        shared_array[:] = frame_hires

    # logger.info("frame hires shape: ", frame_hires.shape)

    frame_lores = cv2.resize(
        frame_hires,
        None,
        fx=1 / preview_downscale_factor,
        fy=1 / preview_downscale_factor,
        interpolation=cv2.INTER_NEAREST,
    )

    future: Future | None = None
    if frame_lores is not None and frame_lores.size:
        has_movement, mask = motion_detector.is_moving(frame_lores)

        with tracker_lock:
            is_tracking = tracking.is_set()

        if is_tracking:
            is_coasting = untracked_frames_count
            if not is_coasting:
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
                        instantaneous_velocity_vector = (
                            current_pos - last_pos
                        ) / dt
                        velocity_buffer.append(instantaneous_velocity_vector)

                    if len(velocity_buffer) > 0:
                        average_velocity_vector = np.mean(
                            list(velocity_buffer), axis=0
                        )

                        # 4. Calculate speed (magnitude) from the AVERAGE vector
                        smoothed_pixels_per_second = np.linalg.norm(
                            average_velocity_vector
                        )

                        stationary_threshold_pixels_per_sec = 5
                        # 5. Apply the stationary threshold
                        if (
                            smoothed_pixels_per_second
                            < stationary_threshold_pixels_per_sec
                        ):
                            logger.info(
                                "SNAPPING SPEED TO ZEEEEEEEEEEEEEEEROOOOOOOOOO: %d",
                                smoothed_pixels_per_second,
                            )
                            final_speed_pixels_per_sec = (
                                0.0  # Snap to zero if it's just jitter
                            )
                        else:
                            final_speed_pixels_per_sec = smoothed_pixels_per_second

                        logger.info(
                            f"Smoothed Speed: {final_speed_pixels_per_sec:.2f} px/s"
                        )

                    last_known_bbox = (current_time, bbox)
                    output_buffer.append(
                        (
                            0,
                            time.monotonic_ns(),
                            frame_lores,
                            [("tracker", 1, bbox)],
                        )
                    )
                else:
                    untracked_frames_count = 1
                    total_untracked_frames_count += 1
            else:
                total_untracked_frames_count += 1
                untracked_frames_count += 1

                if total_untracked_frames_count >= 90:
                    total_untracked_frames_count = 0
                    untracked_frames_count = 0
                    tracker = None
                    tracking.clear()

                if untracked_frames_count > 30:
                    untracked_frames_count = 0

                output_buffer.append(
                    (
                        0,
                        time.monotonic_ns(),
                        frame_lores,
                        [],
                    )
                )

        else:
            if app_settings.debug_settings.debug_enabled or os.environ.get("DISABLE_AI"):
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
                mask_output_buffer.append(buf_highlighted)
            elif has_movement and len(active_futures) < NUM_AI_WORKERS:
                try:
                    timestamp = time.monotonic_ns()

                    rois = motion_detector.create_rois(mask=mask)
                    if rois:
                        with cache_lock:
                            lowres_frame_cache[timestamp] = frame_lores

                        future: Future | None = process_pool.submit(
                            run_object_detection,
                            shape=frame_hires.shape,
                            dtype=frame_hires.dtype,
                            rois=rois,
                            timestamp=timestamp,
                        )
                        active_futures.append(future)
                        future.add_done_callback(on_done)
                except:
                    traceback.print_exc()
                    raise
            elif not has_movement and not active_futures:
                output_buffer.append((0, 0, frame_lores, []))
    return future


# This class instance will hold the camera frames
class StreamingOutput(io.BufferedIOBase):
    def write(self, buf_hires: bytes) -> None:
        if self.closed:
            raise RuntimeError("Stream is closed!! ")

        if len(active_futures) == NUM_AI_WORKERS:
            return

        buf_hires = cv2.imdecode(
            np.frombuffer(buf_hires, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        process_frame(frame_hires=buf_hires)


input_buffer = StreamingOutput()


def stream_nonblocking():
    try:
        import picamera2
        from libcamera import controls

        picamera_available = True
    except ImportError:
        logger.info("Picamera2 not available: %s", traceback.format_exc())
        picamera_available = False
        picamera2 = None
        controls = None

    mock_video_path = os.environ.get("MOCK_CAMERA_PATH")
    if not picamera_available or mock_video_path or os.environ.get("MOCK_CAMERA"):
        streamer_func = get_file_streamer(video_path=mock_video_path)
    else:

        def stream_camera():
            from picamera2.encoders import MJPEGEncoder
            from picamera2.outputs import FileOutput

            try:
                logger.info("Starting stream in 5 seconds...")
                time.sleep(5)
                picam2 = picamera2.Picamera2()

                # noinspection PyUnresolvedReferences
                noise_reduction_mode = controls.draft.NoiseReductionModeEnum.Fast

                camera_config = picam2.create_video_configuration(
                    # main={"size": (1296, 972)},
                    # main={"size": resolution},
                    # queue=False,
                    controls={
                        "FrameRate": 20,
                        "ColourGains": (1, 1),
                        "NoiseReductionMode": noise_reduction_mode,
                    },
                )
                picam2.configure(camera_config)
                picam2.start_preview(picamera2.Preview.NULL)

                # class MeasuringJpegEncoder(JpegEncoder):
                #     def encode_func(self, request, name):
                #         measure_encode = get_measure("JpegEncoder")
                #         res = super().encode_func(request, name)
                #         measure_encode()
                #         return res
                #
                # class MeasuringMJPEGEncoder(MJPEGEncoder):
                #     def encode(self, request, name):
                #         measure_encode = get_measure("MJPEGEncoder")
                #         res = super().encode(request, name)
                #         measure_encode()
                #         return res

                encoder = MJPEGEncoder()

                picam2.start_recording(encoder, FileOutput(input_buffer))
            except:
                traceback.print_exc()
                raise

            @atexit.register
            def cleanup_camera():
                logger.info("Cleaning up picam2")
                picam2.stop_encoder()
                picam2.stop_recording()
                encoder.stop()
                picam2.close()

        streamer_func = stream_camera
    thread_pool.submit(streamer_func)

def start_ffmpeg(*, video_path, output_width, output_height):
    """Launches the FFmpeg process to output a single high-res stream to stdout."""
    global ffmpeg_process

    command = [
#        "taskset", "-c", "0",
        "ffmpeg",
        "-r", "1",
        "-hwaccel",
        "rkmpp",
        "-hwaccel_output_format",
        "drm_prime",
        "-stream_loop",
        "-1",
        "-i",
        video_path,
        #"-c:v", "copy",
        "-f",
        "rawvideo",
        "-vf",
        f"hwupload,scale_rkrga=w={output_width}:h={output_height}:format=rgb24,hwdownload",
        #"-threads:v", "2",
        "-pix_fmt",
        "rgb24",
        "-an",
        "-sn",
        "-dn",
        "-",  # Output to stdout
    ]
    if not platform.machine().lower() == "aarch64":
        command = [
            "ffmpeg",
            "-stream_loop",
            "-1",
            "-i",
            video_path,
            "-f",
            "rawvideo",
            "-an",
            "-sn",
            "-dn",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    logger.info("Starting FFmpeg with command: %s", " ".join(command))
    ffmpeg_process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    def log_stderr():
        if ffmpeg_process.stderr:
            full_error = ""
            for line in iter(ffmpeg_process.stderr.readline, b""):
                full_error += line.decode("utf-8").strip()
            logger.error("FFMPEG: %s", full_error)

    threading.Thread(target=log_stderr, daemon=True).start()
    return ffmpeg_process

def get_file_streamer(video_path: str | None):
    if not video_path:
        video_path = "/home/martin/Downloads/853889-hd_1280_720_25fps.mp4"
        # video_path = "/home/martin/Downloads/4039116-uhd_3840_2160_30fps.mp4"
        # video_path = "/home/martin/Downloads/cat.mov"
        video_path = "/home/martin/Downloads/VID_20250731_093415.mp4"
        video_path = "/mnt/c/Users/mbo20/Downloads/16701023-hd_1920_1080_60fps.mp4"
        # video_path = "/mnt/c/Users/mbo20/Downloads/20522838-hd_1080_1920_30fps.mp4"
        # video_path = "/mnt/c/Users/mbo20/Downloads/13867923-uhd_3840_2160_30fps.mp4"
        # video_path = "/home/martin/Downloads/VID_20250808_080717.mp4"
        # video_path = "/home/martin/Downloads/output_converted.mov"
        # video_path = "/home/martin/Downloads/output_file.mov"
        # video_path = "/home/martin/Downloads/gettyimages-1382583689-640_adpp.mp4"

    def stream_file_hw():
        global double_buffer

        if not os.path.isfile(video_path):
            raise ValueError("File not found: ", video_path)

        logger.info("Starting videostream in 3s...")
        # this sleep is absolutely crucial!!! if we have a low-res video, opencv would be ready before django is and that breaks everything
        time.sleep(3)
    
        print("------------!!!!!!!!!!!!! STREAM")
        high_res_w, high_res_h = 1920, 1080
        channels = 3

        # Pre-fill the work token queue
        for _ in range(NUM_AI_WORKERS):
            if not work_token_queue.full():
                work_token_queue.put(None)

        double_buffer = DoubleBuffer(
            name="hires", shape=(high_res_h, high_res_w, channels), dtype=np.uint8
        )

        process = start_ffmpeg(video_path=video_path, output_width=high_res_w, output_height=high_res_h)

        producer_thread = threading.Thread(
            target=frame_producer, args=(process.stdout, double_buffer), daemon=True
        )
        producer_thread.start()

        logger.info("Waiting for producer to fill first buffer...")
        time.sleep(2)
        logger.info("Consumer loop starting.")

        while True:
            # 1. Wait efficiently for a free worker. Consumes zero CPU.
            work_token_queue.get()

            # 2. Get the latest HIGH-RES frame.
            high_res_frame = double_buffer.read_and_swap()

            f = process_frame(high_res_frame)
            if not f:
                work_token_queue.put(None)

    return stream_file_hw


# This single instance will be imported by other parts of the app


@atexit.register
def cleanup():
    logger.info("[DJANGO SHUTDOWN] Stopping processes.....")

    # with input_buffer.condition:
    input_buffer.close()

    thread_pool.shutdown(wait=True, cancel_futures=True)
    process_pool.shutdown(wait=True, cancel_futures=True)
    cleanup_shared_memory()

    logger.info("[DJANGO SHUTDOWN] Processes stopped..")
