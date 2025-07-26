import time

from django.http.response import StreamingHttpResponse
from django.shortcuts import render
from PIL import Image, ImageFont, ImageDraw
import io
from .utils import stream
from django.utils.timezone import now


# Create your views here.
def index(request):
    return render(request, "core/index.html")


def process_results():
    # --- Result Harvesting Logic ---
    # Check for completed tasks without blocking.
    done_futures = [f for f in stream.active_futures if f.done()]
    for future in done_futures:
        try:
            # .result() will not block since we know the future is done.
            worker_pid, timestamp, result = future.result()
            print(f"Last result from worker {worker_pid} at {timestamp}: {result}")
            # last_results[worker_pid] = result
        except Exception as e:
            print(f"A worker process failed: {e}")
        # Remove the completed future from the active list.
        stream.active_futures.remove(future)


def gen_frames():
    """Video streaming generator function."""

    font_size = 36
    font = ImageFont.load_default(size=font_size)

    img = Image.new("RGB", (640, 480), color="gray")
    draw = ImageDraw.Draw(img)

    while True:
        process_results()

        if stream.camera:
            with stream.output.condition:
                stream.output.condition.wait()
                frame = stream.output.frame
        else:
            text = f"Hello world!\nTime: {now().strftime('%H:%M:%S')}"

            draw.rectangle((0, 0, 640, 480), fill="gray")
            draw.text((0, 0), text, font=font, fill="white")

            # 5. Save the image to an in-memory buffer as a JPEG

            buffer = io.BytesIO()

            img.save(buffer, format="JPEG")

            frame: bytes = buffer.getvalue()
            stream.output.write(buffer.getvalue())
            time.sleep(1)  # Simulate 10 FPS```

        yield b"--frame\nContent-Type: image/jpeg\n\n" + frame + b"\n"


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
