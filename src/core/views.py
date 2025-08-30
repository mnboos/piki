import time

from django.http.response import StreamingHttpResponse
from django.shortcuts import render, redirect

from .utils import clamp
from .utils.shared import (
    settings,
    live_stream_enabled,
    setup_motion_detector,
    app_settings,
    cv2,
)
from .utils import shared


def index(request):
    request.session["prob_threshold"] = int(round(shared.prob_threshold.value, 2) * 100)
    return render(request, "core/index.html", {"servo_range": list(range(-90, 90))})


def config(request):
    request.session.setdefault(
        "mask_transparency", shared.mask_transparency.value * 100
    )
    request.session.setdefault(
        "mog2_history", settings.foreground_mask_options.mog2_history.value
    )
    request.session.setdefault(
        "mog2_var_threshold",
        settings.foreground_mask_options.mog2_var_threshold.value,
    )
    request.session.setdefault(
        "denoise_strength",
        settings.foreground_mask_options.denoise_kernelsize.value,
    )

    if request.method == "POST":
        mask_transparency = int(request.POST.get("mask_transparency"))
        shared.mask_transparency.value = mask_transparency / 100
        request.session["mask_transparency"] = mask_transparency

        mog2_history = int(request.POST.get("mog2_history"))
        settings.foreground_mask_options.mog2_history.value = mog2_history
        request.session["mog2_history"] = mog2_history

        mog2_var_threshold = int(request.POST.get("mog2_var_threshold"))
        settings.foreground_mask_options.mog2_var_threshold.value = mog2_var_threshold
        request.session["mog2_var_threshold"] = mog2_var_threshold

        denoise_strength = int(request.POST.get("denoise_strength"))
        if denoise_strength != 0 and not denoise_strength % 2:
            denoise_strength += 1

        settings.foreground_mask_options.denoise_kernelsize.value = denoise_strength
        request.session["denoise_strength"] = denoise_strength

        setup_motion_detector()

    return render(request, "core/config.html", {})


def stream_camera():
    """Video streaming generator function with corrected drawing logic."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = (255, 255, 255)  # White in BGR
    box_color = (0, 255, 128)  # A nice green for the boxes
    thickness = 2

    try:
        app_settings.debug_settings.debug_enabled = False
        live_stream_enabled.set()
        shared.is_object_detection_disabled.clear()
        while True:
            # This is an efficient way to wait for new frames without burning CPU
            shared.output_buffer.wait_for_data()
            if shared.output_buffer.queue.empty():
                continue

            latest_result = shared.output_buffer.popleft()
            if latest_result is None:
                time.sleep(0.01)
                continue

            # The 'frame' here is the low-resolution preview frame
            _worker_pid, _timestamp, frame, detected_objects = latest_result

            # Get the dimensions of the frame we are drawing on.
            frame_height, frame_width, _ = frame.shape

            for label, confidence, bbox in detected_objects:
                # 1. Unpack the NORMALIZED coordinates [ymin, xmin, ymax, xmax]
                #    These are proportional and work for any frame size.
                left, top, w, h = bbox
                left = int(left)
                top = int(top)
                right = int(left + w)
                bottom = int(top + h)

                # # 2. Clamp values to the [0.0, 1.0] range to prevent errors
                # ymin = max(0.0, ymin)
                # xmin = max(0.0, xmin)
                # ymax = min(1.0, ymax)
                # xmax = min(1.0, xmax)
                #
                # # 3. Denormalize to get PIXEL coordinates for the CURRENT frame
                # left = int(xmin * frame_width)
                # top = int(ymin * frame_height)
                # right = int(xmax * frame_width)
                # bottom = int(ymax * frame_height)

                # 4. Draw the bounding box using the calculated pixel coordinates
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness)

                # 5. Prepare and draw the text label
                text_to_draw = f"{label} ({confidence:.1%})"

                # Create a solid background for the text for better readability
                (text_w, text_h), _ = cv2.getTextSize(
                    text_to_draw, font, font_scale, thickness
                )
                text_bg_rect_start = (left, top - text_h - 7)
                text_bg_rect_end = (left + text_w, top)
                cv2.rectangle(
                    frame, text_bg_rect_start, text_bg_rect_end, box_color, -1
                )  # -1 thickness for filled rectangle

                # Position text on top of the background
                cv2.putText(
                    frame,
                    text_to_draw,
                    (left, top - 5),  # Position text inside the background
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text for contrast
                    1,
                    cv2.LINE_AA,
                )

            # Convert the processed RGB frame to BGR for web streaming and encode
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".jpeg", bgr)
            if success:
                frame_bytes = buffer.tobytes()
                yield b"--frame\nContent-Type: image/jpeg\n\n" + frame_bytes + b"\n"
    finally:
        print("disable livestream")
        live_stream_enabled.clear()


def stream_mask():
    try:
        # shared.is_mask_streaming_enabled.set()
        shared.is_object_detection_disabled.set()
        shared.live_stream_enabled.clear()
        app_settings.debug_settings.debug_enabled = True
        while True:
            time.sleep(0.01)
            if shared.mask_output_buffer.queue.empty():
                continue

            frame = shared.mask_output_buffer.popleft()
            if frame is None or not frame.size:
                continue

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
            success, buffer = cv2.imencode(".jpg", frame, encode_param)
            if success:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
    finally:
        # shared.is_mask_streaming_enabled.clear()
        shared.is_object_detection_disabled.clear()
        app_settings.debug_settings.debug_enabled = False


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
        shared.prob_threshold.value = threshold / 100.0
    return redirect("index")


def move_servo(request):
    """Route to handle servo movement from form submission."""
    if request.method == "POST":
        angle = int(request.POST.get("tilt_position", 0))
        request.session["tilt_position"] = clamp(angle, minimum=-90, maximum=90)

        angle = int(request.POST.get("pan_position", 0))
        request.session["pan_position"] = clamp(angle, minimum=-90, maximum=90)
    return redirect("index")
