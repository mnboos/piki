import concurrent.futures
import dataclasses
import os
from concurrent import futures
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Condition, Event
from collections import deque
import multiprocessing as mp
import io
import time
import atexit
from concurrent.futures import ProcessPoolExecutor, Future, ThreadPoolExecutor
from threading import Thread
import cv2
import numpy as np


NUM_AI_WORKERS: int = 1

active_futures: list[Future] = []
live_stream_enabled = Event()

max_output_timestamp = 0
output_buffer = deque(maxlen=50)

process_pool = ProcessPoolExecutor(max_workers=NUM_AI_WORKERS)
thread_pool = ThreadPoolExecutor(max_workers=1)

# A list to keep track of tasks that have been submitted but are not yet complete.


@dataclasses.dataclass
class DetectionInput:
    foreground_mask: np.ndarray
    current_frame: np.ndarray


def run_object_detection(frame_data, timestamp):
    """
    Symbolic AI function. It gets the raw frame data, processes it,
    and returns its result along with its unique process ID.
    """

    from .ai import detect_objects

    worker_pid = mp.current_process().pid
    result = detect_objects(frame_data)
    return worker_pid, timestamp, frame_data, result


def on_done(future: Future):
    global max_output_timestamp

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

    def write(self, buf: np.ndarray) -> None:
        with self.condition:
            if self.closed:
                raise RuntimeError("Stream is closed")

            self.frame = buf[:]

        # foreground_mask = self.backSub.apply(self.frame)
        if len(active_futures) < NUM_AI_WORKERS:
            try:
                timestamp = time.monotonic_ns()

                global max_output_timestamp

                future: Future = process_pool.submit(
                    run_object_detection,
                    frame_data=buf[:],
                    timestamp=timestamp,
                )

                active_futures.append(future)
                future.add_done_callback(on_done)

            except RuntimeError as e:
                print("ProcessPoolExecutor unusable")
                raise KeyboardInterrupt from e


def __setup_cam():
    picam2 = None
    stream_output = StreamingOutput()
    try:
        from picamera2 import Picamera2, Preview
        from picamera2.encoders import JpegEncoder
        from picamera2.outputs import FileOutput, CircularOutput

        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"size": (640, 480)}, controls={"ColourGains": (1, 1)}
        )
        picam2.configure(camera_config)

        picam2.start_preview(Preview.NULL)
        picam2.start_recording(JpegEncoder(num_threads=1), FileOutput(stream_output))

    except Exception as e:
        print(f"Hardware initialization failed: {e}")

        def stream_file(video_path: str):
            if not os.path.isfile(video_path):
                raise ValueError("File not found: ", video_path)

            # this sleep is absolutely crucial!!! if we have a low-res video, opencv would be ready before django is and that breaks everything
            time.sleep(3)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Error: Could not open video at {video_path}")

            def close_video():
                cap.release()

            atexit.register(close_video)

            # backSub = cv2.createBackgroundSubtractorMOG2()
            backSub = cv2.createBackgroundSubtractorMOG2(
                history=10000, varThreshold=8, detectShadows=False
            )

            while True:
                if stream_output.closed:
                    break

                max_fps = 30
                time.sleep(1 / max_fps)
                try:
                    ret, frame = cap.read()

                    if not ret:
                        # If it ended, rewind to the first frame (frame 0)
                        print("Video stream ended. Rewinding.")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue  # Skip the rest of this iteration and try reading the new first frame
                    else:
                        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # frame = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)

                        resized_frame = cv2.resize(frame, (640, 480))
                        # bgr = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
                        # resized_frame = cv2.resize(
                        #     resized_frame,
                        #     None,
                        #     fx=2,
                        #     fy=2,
                        #     interpolation=cv2.INTER_LINEAR,
                        # )
                        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                        foreground_mask = backSub.apply(gray)
                        foreground_mask = cv2.cvtColor(
                            foreground_mask, cv2.COLOR_GRAY2BGR
                        )

                        # resized_frame = cv2.resize(frame, (1024, 768))
                        # resized_frame = cv2.resize(
                        #     frame, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR
                        # )
                        #
                        # w = 640
                        # h = 480
                        #
                        # center = resized_frame.shape
                        # x = center[1] / 2 - w / 2
                        # y = center[0] / 2 - h / 2
                        #
                        # crop_img = resized_frame[
                        #     int(y) : int(y + h), int(x) : int(x + w)
                        # ]

                        # _, arr = cv2.imencode(".jpg", foreground_mask)
                        # _, arr = cv2.imencode(".jpg", resized_frame)
                        # stream_output.write(arr.tobytes())
                        stream_output.write(foreground_mask)
                except KeyboardInterrupt:
                    print("shutting down here!!!")
                    break
            print("Stopping filestream...")
            cap.release()

        # video_path = "/home/martin/Downloads/cat.mov"
        video_path = "/home/martin/Downloads/output_converted.mov"
        # video_path = "/home/martin/Downloads/output_file.mov"
        # video_path = "/home/martin/Downloads/gettyimages-1382583689-640_adpp.mp4"

        if not os.path.isfile(video_path):
            raise RuntimeError(f"File not found: {video_path}")

        thread_pool.submit(stream_file, video_path=video_path)
        # stream_file(video_path)
        print("Streaming video from file")

    return picam2, stream_output


time.sleep(3)
# This single instance will be imported by other parts of the app
_camera, input_buffer = __setup_cam()


def cleanup():
    print("[DJANGO SHUTDOWN] Stopping processes...")
    if _camera:
        print("Stopping camera and cleaning up GPIO")
        _camera.stop_recording()
        _camera.close()

    with input_buffer.condition:
        input_buffer.close()

        process_pool.shutdown(wait=True, cancel_futures=True)
        thread_pool.shutdown(wait=True, cancel_futures=True)

    print("[DJANGO SHUTDOWN] Processes stopped.")


atexit.register(cleanup)
