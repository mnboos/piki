import time

import cv2
from django.http.response import StreamingHttpResponse
from django.shortcuts import render, redirect
from .utils import clamp
from .utils import stream, ai


# Create your views here.
def index(request):
    request.session["prob_threshold"] = int(round(ai.prob_threshold.value, 2) * 100)

    return render(request, "core/index.html", {"servo_range": list(range(-90, 90))})


def gen_frames():
    """Video streaming generator function, converted to pure OpenCV."""

    # --- OpenCV Font and Color Configuration ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = (255, 255, 255)  # White in BGR
    rect_color = (0, 0, 255)  # Red in BGR
    thickness = 2

    print("Starting OpenCV video retrieval...")
    try:
        stream.live_stream_enabled.set()

        while True:
            time.sleep(0.01)
            # Get data from the worker output buffer

            # The 'frame' variable is expected to be a NumPy array
            if stream.output_buffer.queue.empty():
                continue

            latest_result = stream.output_buffer.popleft()
            # print("latest result: ", latest_result, flush=True)
            if not latest_result:
                continue
            # print("res: ", latest_result)
            worker_pid, timestamp, frame, detected_objects = latest_result



            # This remains an empty list as per your original non-commented code.
            # If you were to find contours, you would do it here on the 'frame'.
            contours = []

            # --- Drawing Logic using OpenCV ---
            # All drawing happens directly on the 'frame' NumPy array.

            # Loop through contours (currently does nothing as 'contours' is empty)
            for cnt in contours:
                rect = cv2.boundingRect(cnt)
                x, y, w, h = rect
                min_size = 20
                if min_size <= h <= w:
                    # Draw a red rectangle using OpenCV
                    cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, thickness)

            # Loop through objects detected by the AI model
            for label, confidence, bbox in detected_objects:
                # Assuming bbox has .x and .y attributes for the top-left corner
                # For drawing text above the box, you might use (bbox.x, bbox.y - 10)
                x, y, _, _ = bbox
                text_position = (int(x), int(y))

                text_to_draw = f"{label} ({confidence:.2%})"

                # Draw the text using OpenCV
                cv2.putText(
                    frame,
                    text_to_draw,
                    text_position,
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )

            # --- Final Encoding and Yielding ---
            # Encode the modified frame (with drawings) to JPEG format in memory
            success, buffer = cv2.imencode(".jpg", frame)

            # If encoding was successful, yield the frame
            if success:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )

    finally:
        # This part remains the same
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
