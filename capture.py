
from picamera2 import Picamera2, Preview

def main():
    picam2 = Picamera2()
    picam2.start_preview(None)
    picam2.start(show_preview=False)
    picam2.capture_file("photo_now.jpg")


if __name__ == "__main__":
    main()
