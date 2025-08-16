import os
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import shared_memory, Lock
import multiprocessing as mp
import io
import time
import atexit
from concurrent.futures import ProcessPoolExecutor, Future, ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import cv2
import numpy as np
import traceback

from .func import (
    get_padded_roi_images,
)
from .metrics import LiveMetricsDashboard
from .shared import (
    mask_transparency,
    is_mask_streaming_enabled,
    output_buffer,
    mask_output_buffer,
    live_stream_enabled,
    motion_detector,
    NUM_AI_WORKERS,
    preview_downscale_factor,
    ai_input_size,
    is_object_detection_disabled,
)


SHM_NAME = "psm_frame_buffer"  # A unique name for our shared memory block
shm_lock = Lock()  # To synchronize access to the shared memory
shared_mem: SharedMemory | None = None  # Will hold the SharedMemory instance
shared_array: Optional[np.typing.NDArray] = (
    None  # The numpy array view of the shared memory
)


def setup_shared_memory(frame_shape, frame_dtype):
    global shared_mem
    global shared_array

    """Creates the shared memory block based on the first frame's properties."""
    try:
        # Create a new shared memory block
        size = int(np.prod(frame_shape) * np.dtype(frame_dtype).itemsize)
        shared_mem = SharedMemory(create=True, size=size, name=SHM_NAME)
        print(
            f"Created shared memory block '{SHM_NAME}' with size {size / 1024**2:.2f} MB"
        )
    except FileExistsError:
        # If it already exists from a previous crashed run, unlink it and retry
        print("Shared memory block already exists, unlinking and recreating.")
        SharedMemory(name=SHM_NAME).unlink()
        size = int(np.prod(frame_shape) * np.dtype(frame_dtype).itemsize)
        shared_mem = SharedMemory(create=True, size=size, name=SHM_NAME)

    # Create a NumPy array that uses the shared memory buffer
    shared_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shared_mem.buf)
    return shared_array


def cleanup_shared_memory():
    """Closes and unlinks the shared memory block on application exit."""
    print("Cleaning up shared memory...")
    if shared_mem:
        shared_mem.close()
        shared_mem.unlink()  # Free the memory block


active_futures: list[Future] = []
max_output_timestamp = 0
dashboard = LiveMetricsDashboard()
process_pool = ProcessPoolExecutor(max_workers=NUM_AI_WORKERS)
thread_pool = ThreadPoolExecutor(max_workers=1)


try:
    import picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
    from libcamera import controls

    PICAMERA_AVAILABLE = True
except ImportError:
    print("Picamera2 not available", traceback.format_exc())
    PICAMERA_AVAILABLE = False
    picamera2 = None
    MJPEGEncoder = None
    FileOutput = None
    controls = None


def get_measure(description: str):
    now = time.perf_counter()

    def measure():
        end = time.perf_counter()
        ms = str(round((end - now) * 1000, 2)).rjust(5)
        print(f"{description}: {ms} ms")

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

        total_duration = 0
        all_detections = []

        from .ai import detect_objects

        for img, scale, effective_origin in padded_images_and_details:
            eff_orig_x, eff_orig_y = effective_origin

            duration, detections = detect_objects(img)
            total_duration += duration

            # 'detections' is a list of (label, confidence, local_pixel_bbox)
            # where local_pixel_bbox is (x, y, w, h)
            for label, confidence, local_pixel_bbox in detections:
                # --- START: Correct Transformation Logic ---

                # 1. Unpack the LOCAL PIXEL coordinates (x, y, w, h) from ai.py
                local_px_x, local_px_y, local_px_w, local_px_h = local_pixel_bbox

                # 2. Convert to (xmin, ymin, xmax, ymax) format
                local_px_xmin = local_px_x
                local_px_ymin = local_px_y
                local_px_xmax = local_px_x + local_px_w
                local_px_ymax = local_px_y + local_px_h

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

        result = total_duration, all_detections

        frame_lores = cv2.resize(
            frame_hires,
            None,
            fx=1 / preview_downscale_factor,
            fy=1 / preview_downscale_factor,
        )

        return worker_pid, timestamp, frame_lores, result

    except Exception as e:
        print(
            f"!!!!!!!!!!!!!! FATAL ERROR IN AI WORKER (PID: {os.getpid()}) !!!!!!!!!!!!!!"
        )
        traceback.print_exc()
        raise e
    finally:
        if existing_shm is not None:
            existing_shm.close()


def on_done(future: Future):
    global max_output_timestamp
    active_futures.remove(future)

    try:
        worker_pid, timestamp, frame, detected_objects = future.result()
        if timestamp >= max_output_timestamp or True:
            max_output_timestamp = timestamp

            if detected_objects:
                inference_time, detections = detected_objects
                dashboard.update(worker_id=worker_pid, inference_time=inference_time)
                if live_stream_enabled.is_set():
                    output_buffer.append((worker_pid, timestamp, frame, detections))
    except BrokenProcessPool:
        print("Pool already broken, when future was done. Shutting down...")
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Future done, shutting down...")
    except:
        traceback.print_exc()
        raise


def process_frame(frame_hires: np.ndarray):
    global shared_array

    # --- Shared Memory Logic ---
    # On the first run, create the shared memory block

    # Lock the shared memory, copy the new frame data into it, and then unlock.
    # This ensures a worker process doesn't read a partially updated frame.
    with shm_lock:
        if shared_array is None:
            shared_array = setup_shared_memory(frame_hires.shape, frame_hires.dtype)

        shared_array[:] = frame_hires

    frame_lores = cv2.resize(
        frame_hires,
        None,
        fx=1 / preview_downscale_factor,
        fy=1 / preview_downscale_factor,
    )

    if frame_lores is not None and frame_lores.size:
        has_movement, mask = motion_detector.is_moving(frame_lores)
        if is_mask_streaming_enabled.is_set():
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

                global max_output_timestamp

                rois = motion_detector.create_rois(mask=mask)

                future: Future = process_pool.submit(
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
            # print("not detecting on current frame")
            output_buffer.append((0, 0, frame_lores, []))


# This class instance will hold the camera frames
class StreamingOutput(io.BufferedIOBase):
    def write(self, buf_hires: bytes) -> None:
        if self.closed:
            raise RuntimeError("Stream is closed")

        if len(active_futures) == NUM_AI_WORKERS:
            return

        buf_hires = cv2.imdecode(
            np.frombuffer(buf_hires, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        process_frame(frame_hires=buf_hires)


input_buffer = StreamingOutput()


def stream_camera():
    try:
        print("Starting stream in 5 seconds...")
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
        print("Cleaning up picam2")
        picam2.stop_encoder()
        picam2.stop_recording()
        encoder.stop()
        picam2.close()


def stream_nonblocking():
    mock_video_path = os.environ.get("MOCK_CAMERA_VIDEO_PATH")
    if not PICAMERA_AVAILABLE or mock_video_path or os.environ.get("MOCK_CAMERA"):
        streamer_func = get_file_streamer(video_path=mock_video_path)
    else:
        streamer_func = stream_camera
    thread_pool.submit(streamer_func)


def get_file_streamer(video_path: str | None):
    if not video_path:
        video_path = "/home/martin/Downloads/853889-hd_1280_720_25fps.mp4"
        # video_path = "/home/martin/Downloads/4039116-uhd_3840_2160_30fps.mp4"
        # video_path = "/home/martin/Downloads/cat.mov"
        video_path = "/home/martin/Downloads/VID_20250731_093415.mp4"
        # video_path = "/mnt/c/Users/mbo20/Downloads/16701023-hd_1920_1080_60fps.mp4"
        # video_path = "/mnt/c/Users/mbo20/Downloads/20522838-hd_1080_1920_30fps.mp4"
        # video_path = "/mnt/c/Users/mbo20/Downloads/13867923-uhd_3840_2160_30fps.mp4"
        # video_path = "/home/martin/Downloads/VID_20250808_080717.mp4"
        # video_path = "/home/martin/Downloads/output_converted.mov"
        # video_path = "/home/martin/Downloads/output_file.mov"
        # video_path = "/home/martin/Downloads/gettyimages-1382583689-640_adpp.mp4"

    def stream_file():
        if not os.path.isfile(video_path):
            raise ValueError("File not found: ", video_path)

        print("Starting videostream in 3s...")
        # this sleep is absolutely crucial!!! if we have a low-res video, opencv would be ready before django is and that breaks everything
        time.sleep(3)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Error: Could not open video at {video_path}")

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        sleep_time = round(1 / fps, 4)
        print("FPS: ", fps, sleep_time)

        @atexit.register
        def close_video():
            cap.release()

        last_time = time.perf_counter()

        while True:
            now = time.perf_counter()
            time_passed = now - last_time
            last_time = now

            remaining_sleep_time = max(sleep_time - time_passed, sleep_time)
            time.sleep(remaining_sleep_time)
            try:
                ret, frame = cap.read()

                if not ret:
                    # If it ended, rewind to the first frame (frame 0)
                    print("Video stream ended. Rewinding.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue  # Skip the rest of this iteration and try reading the new first frame
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame_bytes_hires = cv2.imencode(".jpeg", frame)[1].tobytes()
                    # stream_output.write(frame_bytes_hires)
                    process_frame(frame_hires=frame)
            except KeyboardInterrupt:
                print("shutting down here!!!")
                break
            except Exception:
                traceback.print_exc()
                raise
        print("Stopping filestream...")
        cap.release()

    if not os.path.isfile(video_path):
        raise RuntimeError(f"File not found: {video_path}")
    return stream_file


# This single instance will be imported by other parts of the app


@atexit.register
def cleanup():
    print("[DJANGO SHUTDOWN] Stopping processes...")

    # with input_buffer.condition:
    input_buffer.close()

    thread_pool.shutdown(wait=True, cancel_futures=True)
    process_pool.shutdown(wait=True, cancel_futures=True)
    cleanup_shared_memory()

    print("[DJANGO SHUTDOWN] Processes stopped.")
