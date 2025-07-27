from multiprocessing import Condition, Event
import multiprocessing as mp
import io
import time
import atexit
from concurrent.futures import ProcessPoolExecutor


NUM_AI_WORKERS: int = 2

executor = ProcessPoolExecutor(max_workers=NUM_AI_WORKERS)

# A list to keep track of tasks that have been submitted but are not yet complete.
active_futures = []
live_stream_enabled = Event()


def run_object_detection(frame_data):
    """
    Symbolic AI function. It gets the raw frame data, processes it,
    and returns its result along with its unique process ID.
    """

    from .ai import detect_objects

    # print(f"[{time.monotonic_ns()}] run ai on frame: {frame_data}")

    # print("Importing ai...")

    # Get the unique ID of the process doing the work
    worker_pid = mp.current_process().pid
    result = detect_objects(frame_data)

    # print(f"[WORKER {worker_pid}] Processing frame...")
    # time.sleep(0.2)  # Simulate CPU-bound work

    # The worker function MUST return the result.
    # result_text = f"Worker {worker_pid} at {time.strftime('%H:%M:%S')}"
    # print(f"[Worker-{worker_pid}]: result: ", result_text)
    return worker_pid, time.monotonic_ns(), result


def process_results():
    done_futures = [f for f in active_futures if f.done()]
    results = []
    for future in done_futures:
        try:
            # worker_pid, timestamp, result =
            results.append(future.result())
            # print(f"Last result from worker {worker_pid} at {timestamp}: {result}")
        except Exception as e:
            print(f"A worker process failed: {e}")
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
            self.frame = buf
            self.condition.notify_all()

        if len(active_futures) < NUM_AI_WORKERS:
            future = executor.submit(run_object_detection, frame_data=buf)
            active_futures.append(future)

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
            JpegEncoder(num_threads=1), CircularOutput(stream_output)
        )

        def cleanup():
            print("Stopping camera and cleaning up GPIO")
            picam2.stop_recording()
            picam2.close()

        atexit.register(cleanup)

    except Exception as e:
        print(f"Hardware initialization failed: {e}")
    return picam2, stream_output


# This single instance will be imported by other parts of the app
camera, output = __setup_cam()
