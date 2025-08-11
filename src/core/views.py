import time

import cv2
from django.http.response import StreamingHttpResponse
from django.shortcuts import render, redirect
from .utils import clamp
from .utils import stream, ai


def index(request):
    request.session["prob_threshold"] = int(round(ai.prob_threshold.value, 2) * 100)
    return render(request, "core/index.html", {"servo_range": list(range(-90, 90))})


def config(request):
    opt = stream.settings

    request.session.setdefault(
        "mask_transparency", stream.mask_transparency.value * 100
    )
    request.session.setdefault(
        "mog2_history", opt.foreground_mask_options.mog2_history.value
    )
    request.session.setdefault(
        "mog2_var_threshold",
        opt.foreground_mask_options.mog2_var_threshold.value,
    )
    request.session.setdefault(
        "denoise_strength",
        opt.foreground_mask_options.denoise_kernelsize.value,
    )

    if request.method == "POST":
        mask_transparency = int(request.POST.get("mask_transparency"))
        stream.mask_transparency.value = mask_transparency / 100
        request.session["mask_transparency"] = mask_transparency

        mog2_history = int(request.POST.get("mog2_history"))
        opt.foreground_mask_options.mog2_history.value = mog2_history
        request.session["mog2_history"] = mog2_history

        mog2_var_threshold = int(request.POST.get("mog2_var_threshold"))
        opt.foreground_mask_options.mog2_var_threshold.value = mog2_var_threshold
        request.session["mog2_var_threshold"] = mog2_var_threshold

        denoise_strength = int(request.POST.get("denoise_strength"))
        if not denoise_strength % 2:
            denoise_strength += 1

        opt.foreground_mask_options.denoise_kernelsize.value = denoise_strength
        request.session["denoise_strength"] = denoise_strength

        stream.input_buffer.update_motion_detector()

    return render(request, "core/config.html", {})


def stream_camera():
    """Video streaming generator function, converted to pure OpenCV."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = (255, 255, 255)  # White in BGR
    thickness = 2

    try:
        stream.live_stream_enabled.set()
        while True:
            # time.sleep(0.01)
            if stream.output_buffer.queue.empty():
                continue

            latest_result = stream.output_buffer.popleft()
            if not latest_result:
                continue
            worker_pid, timestamp, frame, detected_objects = latest_result

            for label, confidence, bbox in detected_objects:
                x, y, _, _ = bbox
                text_position = (int(x), int(y))
                text_to_draw = f"{label} ({confidence:.2%})"

                cv2.putText(
                    frame,
                    text_to_draw,
                    text_position,
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".jpeg", bgr)
            if success:
                frame_bytes = buffer.tobytes()
                yield (b"--frame\nContent-Type: image/jpeg\n\n" + frame_bytes + b"\n")
    finally:
        stream.live_stream_enabled.clear()


def stream_mask():
    try:
        stream.is_mask_streaming_enabled.set()
        while True:
            time.sleep(0.01)
            if stream.mask_output_buffer.queue.empty():
                continue

            frame = stream.mask_output_buffer.popleft()
            if frame is None or not frame.size:
                continue

            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
    finally:
        print("disable stream")
        # stream.is_object_detection_disabled.clear()
        # stream.is_mask_streaming_enabled.clear()


def video_feed(request):
    """Video streaming route."""
    return StreamingHttpResponse(
        stream_camera(), content_type="multipart/x-mixed-replace; boundary=frame"
    )


def mask_feed(request):
    """Video streaming route."""
    return StreamingHttpResponse(
        stream_mask(), content_type="multipart/x-mixed-replace; boundary=frame"
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
