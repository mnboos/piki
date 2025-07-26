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
    done_futures = [f for f in stream.active_futures if f.done()]
    results = []
    for future in done_futures:
        try:
            # worker_pid, timestamp, result =
            results.append(future.result())
            # print(f"Last result from worker {worker_pid} at {timestamp}: {result}")
        except Exception as e:
            print(f"A worker process failed: {e}")
        stream.active_futures.remove(future)
    return results


def gen_frames():
    """Video streaming generator function."""

    font_size = 24
    font = ImageFont.load_default(size=font_size)

    img = Image.new("RGB", (640, 480), color="gray")
    draw = ImageDraw.Draw(img)

    while True:
        results = process_results()

        if stream.camera:
            with stream.output.condition:
                stream.output.condition.wait()
                # frame = stream.output.frame
        else:
            text = f"Time: {now().strftime('%H:%M:%S')}"

            draw.rectangle((0, 0, 640, 480), fill="gray")
            draw.text((0, 0), text, font=font, fill="white")

            buffer = io.BytesIO()

            img.save(buffer, format="JPEG")

            # frame: bytes = buffer.getvalue()
            stream.output.write(buffer.getvalue()[:])
            time.sleep(1)  # Simulate 10 FPS```

        for r in results:
            worker_pid, timestamp, inference_result = r
            frame, detected_objects = inference_result
            with Image.open(io.BytesIO(frame)) as img:
                draw = ImageDraw.Draw(img)
                for confidence, label, bbox in detected_objects:
                    draw.text(
                        (bbox.x, bbox.y),
                        f"{label} ({confidence:.2%})",
                        font=font,
                        fill="white",
                    )
                if not detected_objects:
                    draw.text((0, 50), "No objects detected", font=font, fill="white")

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")

                yield (
                    b"--frame\nContent-Type: image/jpeg\n\n" + buffer.getvalue() + b"\n"
                )
        # yield b"--frame\nContent-Type: image/jpeg\n\n" + frame + b"\n"


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
