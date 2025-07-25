from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from gpiozero import Servo

import io
from threading import Condition
import atexit


def setup_cam():
    # --- Servo and Camera Setup ---
    # It's recommended to use a try/except block for hardware initialization
    picam2 = None
    output = None
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

        class StreamingOutput(io.BufferedIOBase):
            def __init__(self):
                self.frame = None
                self.condition = Condition()

            def write(self, buf):
                with self.condition:
                    self.frame = buf
                    self.condition.notify_all()

        output = StreamingOutput()
        picam2.start_recording(JpegEncoder(), FileOutput(output))

        # --- Cleanup ---
        def cleanup():
            print("Stopping camera and cleaning up GPIO")
            picam2.stop_recording()

        atexit.register(cleanup)

    except Exception as e:
        print(f"Hardware initialization failed: {e}")
        # Handle the error gracefully, maybe set a flag to show an error on the webpage
        servo = None
        picam2 = None
    return picam2, output
