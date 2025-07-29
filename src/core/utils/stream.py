import concurrent.futures
import os
from multiprocessing import Condition, Event
import multiprocessing as mp
import io
import time
import atexit
from concurrent.futures import ProcessPoolExecutor, Future
from threading import Thread
import cv2

NUM_AI_WORKERS: int = 2

active_futures: list[Future] = []
live_stream_enabled = Event()

try:
    executor = ProcessPoolExecutor(max_workers=NUM_AI_WORKERS)

    # A list to keep track of tasks that have been submitted but are not yet complete.

    def run_object_detection(frame_data):
        """
        Symbolic AI function. It gets the raw frame data, processes it,
        and returns its result along with its unique process ID.
        """
        from .ai import detect_objects

        worker_pid = mp.current_process().pid
        result = detect_objects(frame_data)
        return worker_pid, time.monotonic_ns(), result

    def process_results():
        done_futures = [f for f in active_futures if f.done()]
        results = []
        future: Future
        for future in done_futures:
            try:
                # worker_pid, timestamp, result =
                result = future.result()
                # print("result: ", result)
                results.append(result)
                # print(f"Last result from worker {worker_pid} at {timestamp}: {result}")
            except Exception as e:
                print(f"A worker process failed: {e}")
                raise
            except KeyboardInterrupt:
                print("shutting down here!!!")
                break
            active_futures.remove(future)
        return results

    # This class instance will hold the camera frames
    class StreamingOutput(io.BufferedIOBase):
        def __init__(self) -> None:
            self.frame: bytes = b""
            self.condition = Condition()
            print("StreamingOutput created")

        def write(self, buf: bytes) -> None:
            with self.condition:
                # if self.closed:
                #     raise RuntimeError("StreamingOutput closed")

                self.frame = buf
                # self.condition.notify_all()

                if not self.closed and len(active_futures) < NUM_AI_WORKERS:
                    try:
                        future = executor.submit(run_object_detection, frame_data=buf)
                        active_futures.append(future)
                    except RuntimeError as e:
                        print("ProcessPoolExecutor unusable")
                        raise KeyboardInterrupt from e

                if not live_stream_enabled.is_set():
                    process_results()

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
            picam2.start_recording(
                JpegEncoder(num_threads=1), FileOutput(stream_output)
            )

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
                    try:
                        # cap = cv2.VideoCapture(video_path)
                        # while cap.isOpened():
                        time.sleep(0.01)
                        ret, frame = cap.read()
                        # if frame is read correctly ret is True

                        # 3. Check if the video has ended
                        if not ret:
                            if output.closed:
                                break

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

            thread = Thread(
                target=stream_file,
                kwargs={"video_path": "/home/martin/Downloads/cat.mov"},
                daemon=True,
            )
            print("Streaming video from file")
            thread.start()
        return picam2, stream_output

    # This single instance will be imported by other parts of the app
    _camera, output = __setup_cam()

    def cleanup():
        print("[DJANGO SHUTDOWN] Stopping processes...")
        if _camera:
            print("Stopping camera and cleaning up GPIO")
            _camera.stop_recording()
            _camera.close()

        with output.condition:
            output.close()

            # shutting_down.set()
            concurrent.futures.wait(active_futures)
            executor._shutdown_lock.acquire()
            executor.shutdown(wait=True, cancel_futures=True)
            executor._shutdown_lock.release()

            # executor.shutdown(wait=True, cancel_futures=True)

        print("[DJANGO SHUTDOWN] Processes stopped.")

    atexit.register(cleanup)
except KeyboardInterrupt:
    exit(1)
