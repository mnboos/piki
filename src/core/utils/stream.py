import dataclasses
import os
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Condition, Event
from collections import deque
import multiprocessing as mp
import io
import time
import atexit
from concurrent.futures import ProcessPoolExecutor, Future, ThreadPoolExecutor
import cv2
import numpy as np
import traceback

from picamera2 import controls
from picamera2.encoders import H264Encoder

NUM_AI_WORKERS: int = 2

active_futures: list[Future] = []
live_stream_enabled = Event()

max_output_timestamp = 0
output_buffer = deque(maxlen=10)

process_pool = ProcessPoolExecutor(max_workers=NUM_AI_WORKERS)
thread_pool = ThreadPoolExecutor(max_workers=1)

# A list to keep track of tasks that have been submitted but are not yet complete.


@dataclasses.dataclass
class DetectionInput:
    foreground_mask: np.ndarray
    current_frame: np.ndarray


def run_object_detection(frame_data: np.ndarray, timestamp: int):
    """
    Symbolic AI function. It gets the raw frame data, processes it,
    and returns its result along with its unique process ID.
    """

    from .ai import detect_objects

    print("detect frame")
    worker_pid = mp.current_process().pid
    result = detect_objects(frame_data)
    return worker_pid, timestamp, frame_data, result


def on_done(future: Future):
    global max_output_timestamp

    print("future is done")
    active_futures.remove(future)

    try:
        worker_pid, timestamp, frame, detected_objects = future.result()
        if timestamp >= max_output_timestamp:
            max_output_timestamp = timestamp

            if live_stream_enabled.is_set():
                output_buffer.append(future.result())
    except BrokenProcessPool as ex:
        print("Pool already broken, when future was done. Shutting down...")
    except KeyboardInterrupt:
        print("Future done, shutting down...")


# This class instance will hold the camera frames
class StreamingOutput(io.BufferedIOBase):
    def __init__(self) -> None:
        self.frame: bytes = b""
        self.condition = Condition()
        # self.backSub = cv2.createBackgroundSubtractorMOG2()
        print("StreamingOutput created")
        self.motion_detector = MotionDetector()

    def write(self, buf: bytes) -> None:
        print(f"[{time.monotonic_ns()}]: got frame")
        with self.condition:
            if self.closed:
                raise RuntimeError("Stream is closed")

            buf = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
            buf: np.ndarray
            self.frame = buf

        if buf is not None and buf.size:
            cv2.resize(buf, (640, 480), buf)
            has_movement = self.motion_detector.is_moving(buf)
            if has_movement:
                final_frame = self.motion_detector.highlight_movement_on(buf)
            else:
                final_frame = buf

            print(
                f"Has movement: {has_movement}, active futures: {len(active_futures)}"
            )

            # foreground_mask = self.backSub.apply(self.frame)
            if has_movement and len(active_futures) < NUM_AI_WORKERS:
                try:
                    timestamp = time.monotonic_ns()

                    global max_output_timestamp

                    future: Future = process_pool.submit(
                        run_object_detection,
                        frame_data=buf[:],
                        timestamp=timestamp,
                    )
                    # future: Future = process_pool.submit(
                    #     run_object_detection,
                    #     frame_data=buf[:],
                    #     timestamp=timestamp,
                    # )

                    active_futures.append(future)
                    future.add_done_callback(on_done)

                except RuntimeError as e:
                    traceback.print_exc()
                    raise KeyboardInterrupt from e
            elif not has_movement and not active_futures:
                print("not detecting on current frame")
                output_buffer.append((0, 0, final_frame, []))
            else:
                # this is for testing only
                output_buffer.append((0, 0, final_frame, []))


class MotionDetector:
    def __init__(self, pixelcount_threshold: int = 500, denoise: bool = True):
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.foreground_mask: np.ndarray | None = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.pixelcount_threshold = pixelcount_threshold
        self.denoise = denoise

    def moving_pixel_count(self):
        return cv2.countNonZero(self.foreground_mask)

    def is_moving(self, frame: np.ndarray):
        if self.denoise:
            frame = cv2.GaussianBlur(frame, (33, 33), 0)
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, frame)
        self.foreground_mask = self.backSub.apply(frame)
        cv2.morphologyEx(
            self.foreground_mask, cv2.MORPH_OPEN, self.kernel, self.foreground_mask
        )
        return self.moving_pixel_count() >= self.pixelcount_threshold

    def highlight_movement_on(
        self,
        frame: np.ndarray,
        transparency_factor: float = 0.4,
        overlay_color_bgr: tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        colored_overlay = np.full(frame.shape, overlay_color_bgr, dtype=np.uint8)
        blended = cv2.addWeighted(
            frame,
            1 - transparency_factor,
            colored_overlay,
            transparency_factor,
            0,
        )
        return np.where(
            self.foreground_mask[:, :, None] != 0,
            blended,
            frame,
        )


def __setup_cam():
    stream_output = StreamingOutput()

    resolution = (640, 480)

    try:
        from picamera2 import Picamera2, Preview
        from picamera2.encoders import JpegEncoder
        from picamera2.outputs import FileOutput, CircularOutput
        from libcamera import controls

        def stream_camera():
            try:
                print("Starting stream in 5 seconds...")
                time.sleep(5)
                picam2 = Picamera2()
                camera_config = picam2.create_video_configuration(
                    # main={"size": (1296, 972)},
                    main={"size": (640, 480)},
                    # queue=False,
                    controls={
                        "ColourGains": (1, 1),
                        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off,
                    },
                )
                picam2.configure(camera_config)
                picam2.start_preview(Preview.NULL)

                encoder = JpegEncoder(num_threads=1)
                # encoder = H264Encoder(100_000, repeat=True)

                # encoder.output = CircularOutput(file=stream_output, buffersize=10)
                encoder.frame_skip_count = 10
                # encoder.use_hw = True

                picam2.start_recording(encoder, FileOutput(stream_output))
                # picam2.start()
                # if not encoder.running:
                #     picam2.start_encoder(encoder)
            except:
                traceback.print_exc()
                raise

            def cleanup():
                print("Cleaning up picam2")
                picam2.stop_encoder()
                picam2.stop_recording()
                encoder.stop()
                picam2.close()

            atexit.register(cleanup)

        streamer_func = stream_camera

    except ImportError as e:
        print(f"Hardware initialization failed: {e}")
        traceback.print_exc()

        def stream_file(video_path: str):
            if not os.path.isfile(video_path):
                raise ValueError("File not found: ", video_path)

            print("Starting videostream in 3s...")
            # this sleep is absolutely crucial!!! if we have a low-res video, opencv would be ready before django is and that breaks everything
            time.sleep(3)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Error: Could not open video at {video_path}")

            def close_video():
                cap.release()

            atexit.register(close_video)

            # max_fps = 30
            # motion_detector = MotionDetector()

            while True:
                if stream_output.closed:
                    break

                time.sleep(0.015)
                try:
                    ret, frame = cap.read()

                    if not ret:
                        # If it ended, rewind to the first frame (frame 0)
                        print("Video stream ended. Rewinding.")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue  # Skip the rest of this iteration and try reading the new first frame
                    else:
                        frame_resized = cv2.resize(frame, resolution)
                        stream_output.write(frame_resized)
                except KeyboardInterrupt:
                    print("shutting down here!!!")
                    break
                except Exception:
                    traceback.print_exc()
                    raise
            print("Stopping filestream...")
            cap.release()

        # video_path = "/home/martin/Downloads/853889-hd_1280_720_25fps.mp4"
        # video_path = "/home/martin/Downloads/4039116-uhd_3840_2160_30fps.mp4"
        # video_path = "/home/martin/Downloads/cat.mov"
        video_path = "/home/martin/Downloads/VID_20250731_093415.mp4"
        # video_path = "/home/martin/Downloads/output_converted.mov"
        # video_path = "/home/martin/Downloads/output_file.mov"
        # video_path = "/home/martin/Downloads/gettyimages-1382583689-640_adpp.mp4"

        if not os.path.isfile(video_path):
            raise RuntimeError(f"File not found: {video_path}")

        streamer_func = lambda: stream_file(video_path)

    thread_pool.submit(streamer_func)
    # stream_file(video_path)
    # print("Streaming video from file")

    return stream_output


# This single instance will be imported by other parts of the app
input_buffer = __setup_cam()


def cleanup():
    print("[DJANGO SHUTDOWN] Stopping processes...")

    with input_buffer.condition:
        input_buffer.close()

        process_pool.shutdown(wait=True, cancel_futures=True)
        thread_pool.shutdown(wait=True, cancel_futures=True)

    print("[DJANGO SHUTDOWN] Processes stopped.")


atexit.register(cleanup)
