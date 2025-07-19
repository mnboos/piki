from picamera2 import Picamera2, Preview
from datetime import datetime
from signal import pause


def take_photo():
    picam2 = Picamera2()
    picam2.start_preview(None)
    picam2.start(show_preview=False)
    picam2.capture_file(f"{datetime.now():%Y-%m-%d-%H-%M-%S}.jpg")

take_photo()

pause()
