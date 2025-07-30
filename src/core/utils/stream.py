import concurrent.futures
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
from threading import Thread
import cv2
import numpy as np


NUM_AI_WORKERS: int = 3

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
            output_buffer.append(future.result())
    except BrokenProcessPool:
        print("Pool already broken, when future was done. Shutting down...")
    except KeyboardInterrupt:
        print("Future done, shutting down...")


# This class instance will hold the camera frames
class StreamingOutput(io.BufferedIOBase):
    def __init__(self) -> None:
        self.frame: bytes = b""
        self.condition = Condition()
        print("StreamingOutput created")

    def write(self, buf: bytes) -> None:
        with self.condition:
            if self.closed:
                raise RuntimeError("Stream is closed")

            self.frame = buf
            # self.condition.notify_all()

        # if not self.closed and :
        if len(active_futures) < NUM_AI_WORKERS:
            try:
                timestamp = time.monotonic_ns()
                future = process_pool.submit(
                    run_object_detection, frame_data=self.frame, timestamp=timestamp
                )

                active_futures.append(future)
                future.add_done_callback(on_done)
                # future = executor.submit(
                #     run_object_detection, frame_data=DetectionInput(buf, buf)
                # )

            except RuntimeError as e:
                print("ProcessPoolExecutor unusable")
                raise KeyboardInterrupt from e

        # if not live_stream_enabled.is_set():
        #     process_results()


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

            # 1. Open the video capture object ONCE, outside the loop
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video at {video_path}")
                return

            def close_video():
                cap.release()

            atexit.register(close_video)

            while True:
                if input_buffer.closed:
                    break

                try:
                    # cap = cv2.VideoCapture(video_path)
                    # while cap.isOpened():
                    # time.sleep(0.01)
                    ret, frame = cap.read()
                    # if frame is read correctly ret is True

                    # 3. Check if the video has ended
                    if not ret:
                        # If it ended, rewind to the first frame (frame 0)
                        print("Video stream ended. Rewinding.")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue  # Skip the rest of this iteration and try reading the new first frame
                    else:
                        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # resized_frame = cv2.resize(frame, (1024, 768))
                        resized_frame = cv2.resize(
                            frame, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR
                        )

                        w = 640
                        h = 480

                        center = resized_frame.shape
                        x = center[1] / 2 - w / 2
                        y = center[0] / 2 - h / 2

                        crop_img = resized_frame[
                            int(y) : int(y + h), int(x) : int(x + w)
                        ]

                        _, arr = cv2.imencode(".jpg", crop_img)
                        # _, arr = cv2.imencode(".jpg", resized_frame)
                        stream_output.write(arr.tobytes())
                except KeyboardInterrupt:
                    print("shutting down here!!!")
                    break
            print("Stopping filestream...")
            cap.release()

        thread_pool.submit(stream_file, video_path="/home/martin/Downloads/cat.mov")

        print("Streaming video from file")

    return picam2, stream_output


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
