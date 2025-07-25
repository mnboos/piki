import time

from django.http.response import StreamingHttpResponse
from django.shortcuts import render
from PIL import Image, ImageFont, ImageDraw
import io
from .utils import setup_cam
from django.utils.timezone import now


# Create your views here.
def index(request):
    return render(request, "core/index.html")


def gen_frames():
    """Video streaming generator function."""
    _, output = setup_cam()

    font_size = 36
    font = ImageFont.load_default(size=font_size)

    while True:
        if output:
            with output.condition:
                output.condition.wait()
                frame = output.frame
        else:
            text = f"Hello world!\nTime: {now().strftime('%H:%M:%S')}"

            img = Image.new("RGB", (640, 480), color="gray")
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), text, font=font, fill="white")

            # 5. Save the image to an in-memory buffer as a JPEG

            buffer = io.BytesIO()

            img.save(buffer, format="JPEG")
            frame: bytes = buffer.getvalue()
            time.sleep(1)  # Simulate 10 FPS```
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


def video_feed(request):
    """Video streaming route."""
    return StreamingHttpResponse(
        gen_frames(), content_type="multipart/x-mixed-replace; boundary=frame"
    )


def move_servo(request):
    """Route to handle servo movement from form submission."""
    # if request.method == 'POST' and servo:
    #     slider_value = request.POST.get('slider')
    #     if slider_value is not None:
    #         # Convert slider value (-100 to 100) to servo value (-1 to 1)
    #         servo_value = int(slider_value) / 100.0
    #         servo.value = servo_value
    pass
