import time

import cv2
from django.http.response import StreamingHttpResponse
from django.shortcuts import render, redirect
from PIL import Image, ImageFont, ImageDraw
import io
from .utils import clamp
from .utils import stream, ai


# Create your views here.
def index(request):
    request.session["prob_threshold"] = int(round(ai.prob_threshold.value, 2) * 100)

    return render(request, "core/index.html", {"servo_range": list(range(-90, 90))})


def gen_frames():
    """Video streaming generator function."""

    font_size = 24
    font = ImageFont.load_default(size=font_size)

    # img = Image.new("RGB", (640, 480))

    print("Starting video retrieval...")
    try:
        stream.live_stream_enabled.set()
        while True:
            try:
                worker_pid, timestamp, frame, detected_objects = (
                    stream.output_buffer.popleft()
                )
            except IndexError:
                continue

            # with stream.input_buffer.condition:
            #     stream.input_buffer.condition.wait()
            # frame = stream.output.frame
            # else:
            #     text = f"Time: {now().strftime('%H:%M:%S')}"
            #
            #     draw.rectangle((0, 0, 640, 480), fill="gray")
            #     draw.text((0, 0), text, font=font, fill="white")
            #
            #     buffer = io.BytesIO()
            #
            #     img.save(buffer, format="JPEG")
            #
            #     # frame: bytes = buffer.getvalue()
            #     stream.output.write(buffer.getvalue()[:])
            #     time.sleep(1)  # Simulate 10 FPS```

            # results = stream.process_results()
            # for r in results:
            #     worker_pid, timestamp, frame, detected_objects = r
            # Identifying contours from the threshold

            _, cv_frame = cv2.imencode(".jpg", frame)
            reshaped = frame.reshape((480, 640, 3))

            gray = cv2.cvtColor(reshaped, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 50, 255, 0)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # cv_img = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)

            frame_bytes = cv_frame.tobytes()
            # with Image.fromarray(color_coverted) as img:
            with Image.open(io.BytesIO(frame_bytes)) as img:
                draw = ImageDraw.Draw(img)

                for cnt in contours:
                    rect = cv2.boundingRect(cnt)
                    x, y, w, h = rect
                    if w >= 50 and h >= 50:
                        #     if y > 200:  # Disregard items in the top of the picture
                        draw.rectangle((x, y, x + w, y + h))

                for confidence, label, bbox in detected_objects:
                    # x, y, width, height = bbox
                    # draw.rectangle((x, y, x + width + height, y + width + height))
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
    finally:
        stream.live_stream_enabled.clear()


def video_feed(request):
    """Video streaming route."""
    return StreamingHttpResponse(
        gen_frames(), content_type="multipart/x-mixed-replace; boundary=frame"
    )


def config_ai(request):
    if request.method == "POST":
        threshold = float(request.POST.get("prob_threshold"))
        ai.prob_threshold.value = threshold / 100.0
    return redirect("index")


def move_servo(request):
    """Route to handle servo movement from form submission."""
    if request.method == "POST":
        angle = int(request.POST.get("tilt_position", 0))
        request.session["tilt_position"] = clamp(angle, minimum=-90, maximum=90)

        angle = int(request.POST.get("pan_position", 0))
        request.session["pan_position"] = clamp(angle, minimum=-90, maximum=90)
    return redirect("index")
