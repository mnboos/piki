from multiprocessing import Condition, Queue, Process, Event
import io
from typing import List, Optional
from queue import Empty, Full
import time
import atexit

# --- Configuration ---
NUM_AI_WORKERS: int = 2

# --- Shared State (will be initialized in apps.py) ---
frame_queue: Optional[Queue] = None
results_queue: Optional[Queue] = None
stop_event: Optional[Event] = None
workers: List[Process] = []


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

        if frame_queue:
            timestamp = time.monotonic()
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
            try:
                frame_queue.put((timestamp, buf), block=False)
            except Full:
                pass


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
