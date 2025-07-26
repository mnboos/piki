from multiprocessing import Condition
import multiprocessing as mp
import io
import time
import atexit
from concurrent.futures import ProcessPoolExecutor

# --- Configuration ---
NUM_AI_WORKERS: int = 2

executor = ProcessPoolExecutor(max_workers=NUM_AI_WORKERS)

# A list to keep track of tasks that have been submitted but are not yet complete.
active_futures = []

# --- Shared State (will be initialized in apps.py) ---
# frame_queue: Optional[Queue] = None
# results_queue: Optional[Queue] = None
# stop_event: Optional[Event] = None
# workers: List[Process] = []


def run_ai_on_frame(frame_data):
    """
    Symbolic AI function. It gets the raw frame data, processes it,
    and returns its result along with its unique process ID.
    """
    # print(f"[{time.monotonic_ns()}] run ai on frame: {frame_data}")

    print("Importing ai...")

    from .ai import inference

    # Get the unique ID of the process doing the work
    worker_pid = mp.current_process().pid
    result = inference(frame_data)

    # print(f"[WORKER {worker_pid}] Processing frame...")
    time.sleep(0.2)  # Simulate CPU-bound work

    # The worker function MUST return the result.
    result_text = f"Worker {worker_pid} at {time.strftime('%H:%M:%S')}"
    print(f"[Worker-{worker_pid}]: result: ", result_text)
    return worker_pid, time.monotonic_ns(), result_text


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

        # --- Task Submission Logic ---
        # Only submit a new task if there is a free worker.
        if len(active_futures) < NUM_AI_WORKERS:
            print(f"[{time.monotonic()}] Submit frame to worker...")
            # Submit the function to run with the frame data as its argument
            future = executor.submit(run_ai_on_frame, frame_data=buf)
            active_futures.append(future)

        # if frame_queue:
        #     timestamp = time.monotonic()
        #     try:
        #         frame_queue.get_nowait()
        #     except Empty:
        #         pass
        #     try:
        #         frame_queue.put((timestamp, buf), block=False)
        #     except Full:
        #         pass


def __setup_cam():
    # --- Servo and Camera Setup ---
    # It's recommended to use a try/except block for hardware initialization
    picam2 = None
    stream_output = StreamingOutput()
    try:
        from picamera2 import Picamera2
        from picamera2.encoders import JpegEncoder
        from picamera2.outputs import FileOutput

        # Create a Servo object for GPIO pin 17
        # servo = Servo(17, min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000)

        # --- Camera Setup ---
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"size": (640, 480)}, controls={"ColourGains": (1, 1)}
        )
        picam2.configure(camera_config)

        picam2.start_recording(JpegEncoder(), FileOutput(stream_output))

        # --- Cleanup ---
        def cleanup():
            print("Stopping camera and cleaning up GPIO")
            picam2.stop_recording()

        atexit.register(cleanup)

    except Exception as e:
        print(f"Hardware initialization failed: {e}")
    return picam2, stream_output


# This single instance will be imported by other parts of the app
camera, output = __setup_cam()
