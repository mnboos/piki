import asyncio

from django.http import HttpRequest
from django.http.response import StreamingHttpResponse
from ninja import NinjaAPI, PatchDict, Schema

from .utils.shared import OutputResult, app_settings, cv2, is_object_detection_disabled, output_buffer

# api = NinjaAPI(csrf=True, auth=django_auth)
api = NinjaAPI()


async def stream_camera():
    """Video streaming generator function with corrected drawing logic."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    # text_color = (255, 255, 255)  # White in BGR
    box_color = (0, 255, 128)  # A nice green for the boxes
    thickness = 2

    app_settings.debug_settings.debug_enabled = False
    is_object_detection_disabled.clear()
    while True:
        # This is an efficient way to wait for new frames without burning CPU
        if output_buffer.queue.empty():
            await asyncio.sleep(0.01)
            continue

        latest_result = output_buffer.popleft()
        if not isinstance(latest_result, OutputResult):
            await asyncio.sleep(0.01)
            continue

        if latest_result is None:
            await asyncio.sleep(0.01)
            continue

        frame = latest_result.frame_lores

        # Check if frame is None or an empty numpy array
        if frame is None or (hasattr(frame, "size") and frame.size == 0):
            # No frame to show, wait for the next one
            await asyncio.sleep(0.01)
            continue

        # The 'frame' here is the low-resolution preview frame
        # _worker_pid, _timestamp, frame, detected_objects = latest_result
        frame = latest_result.frame_lores

        # Get the dimensions of the frame we are drawing on.
        # frame_height, frame_width, _ = frame.shape

        for detection in latest_result.detections_denormalized:
            # 1. Unpack the NORMALIZED coordinates [ymin, xmin, ymax, xmax]
            #    These are proportional and work for any frame size.
            left, top, w, h = detection.bbox
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
            text_to_draw = f"{detection.label} ({detection.confidence:.1%})"

            # Create a solid background for the text for better readability
            (text_w, text_h), _ = cv2.getTextSize(text_to_draw, font, font_scale, thickness)
            text_bg_rect_start = (left, top - text_h - 7)
            text_bg_rect_end = (left + text_w, top)
            cv2.rectangle(
                frame,
                text_bg_rect_start,
                text_bg_rect_end,
                box_color,
                -1,
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
        # bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr = frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        success, buffer = cv2.imencode(".jpeg", bgr, encode_param)
        if success:
            frame_bytes = buffer.tobytes()
            yield b"--frame\nContent-Type: image/jpeg\n\n" + frame_bytes + b"\n"


BIN_RESPONSE = {
    "responses": {
        200: {
            "description": "OK",
            "content": {
                "multipart/x-mixed-replace; boundary=frame": {"schema": {"type": "string", "format": "binary"}},
            },
        },
    },
}


@api.get("/video_feed", openapi_extra=BIN_RESPONSE)
async def video_feed(request: HttpRequest):
    """Video streaming route."""
    return StreamingHttpResponse(stream_camera(), content_type="multipart/x-mixed-replace; boundary=frame")


class PikiOptions(Schema):
    mode: str


@api.patch("/update_options", response=PikiOptions)
def update_options(request: HttpRequest, options: PatchDict[PikiOptions]):
    return PikiOptions(mode=options.get("mode", "mask"))
