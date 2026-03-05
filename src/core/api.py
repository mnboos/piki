import asyncio

from django.http import HttpRequest
from django.http.response import StreamingHttpResponse
from ninja import NinjaAPI, PatchDict, Schema

from .utils.shared import app_settings, cv2, is_object_detection_disabled, latest_frame, streaming_active

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
    streaming_active.set()
    try:
        last_ts = 0
        while True:
            # Wait for a new frame from the producer thread
            result = await asyncio.to_thread(latest_frame.wait_for_frame, last_ts)
            if result is None:
                await asyncio.sleep(0.01)
                continue

            frame, detections, last_ts = result

            if frame is None or (hasattr(frame, "size") and frame.size == 0):
                continue

            # Draw detections on the frame
            for detection in detections:
                left, top, w, h = detection.bbox
                left, top, w, h = int(left), int(top), int(w), int(h)
                right = left + w
                bottom = top + h

                cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness)

                text_to_draw = f"{detection.label} ({detection.confidence:.1%})"
                (text_w, text_h), _ = cv2.getTextSize(text_to_draw, font, font_scale, thickness)
                text_bg_rect_start = (left, top - text_h - 7)
                text_bg_rect_end = (left + text_w, top)
                cv2.rectangle(frame, text_bg_rect_start, text_bg_rect_end, box_color, -1)

                cv2.putText(
                    frame,
                    text_to_draw,
                    (left, top - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            success, buffer = cv2.imencode(".jpeg", frame, encode_param)
            if success:
                frame_bytes = buffer.tobytes()
                yield b"--frame\nContent-Type: image/jpeg\n\n" + frame_bytes + b"\n"
    finally:
        streaming_active.clear()


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
