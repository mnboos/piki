import dataclasses
import os
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Condition, Event, Queue
import multiprocessing as mp
import io
import time
import atexit
from concurrent.futures import ProcessPoolExecutor, Future, ThreadPoolExecutor
import cv2
import numpy as np
import traceback
from .metrics import LiveMetricsDashboard, retrieve_queue
from .helpers import MultiprocessingDequeue

NUM_AI_WORKERS: int = 2

active_futures: list[Future] = []
live_stream_enabled = Event()

dashboard_queue = retrieve_queue()
dashboard = LiveMetricsDashboard(retrieve_queue())


max_output_timestamp = 0
# output_buffer = Queue(maxsize=10)
output_buffer = MultiprocessingDequeue(queue=Queue(maxsize=10))

process_pool = ProcessPoolExecutor(max_workers=NUM_AI_WORKERS)
thread_pool = ThreadPoolExecutor(max_workers=1)


# class LiveMetrics:
#     def __init__(self) -> None:
#         self.inference = deque(maxlen=1)
#         self.live = Live(self._generate_table(), screen=False, transient=False)
#
#     def update(self, *, inference_time: float) -> None:
#         self.inference.append(inference_time)
#         # with self.live as l:
#         self.live.update(self._generate_table())
#
#     @staticmethod
#     def _generate_table() -> Table:
#         """Draws a Rich Table from the current dashboard_data."""
#         table = Table(title="Multi-Process Inference Monitor")
#         table.add_column("Worker ID", style="cyan")
#         table.add_column("Inference Time (ms)", justify="right", style="magenta")
#         table.add_column("Items Processed", justify="right", style="green")
#
#         dashboard_data = {
#             0: {"inference_time": "N/A", "items": 0},
#             1: {"inference_time": "N/A", "items": 0},
#         }
#
#         for worker_id, data in dashboard_data.items():
#             table.add_row(
#                 f"Worker {worker_id}",
#                 str(data["inference_time"]),
#                 str(data["items"])
#             )
#         return table

# A list to keep track of tasks that have been submitted but are not yet complete.

# metrics = LiveMetrics()

def get_measure(description: str):
    now = time.perf_counter()

    def measure():
        end = time.perf_counter()
        ms = str(round((end - now) * 1000, 2)).rjust(5)
        print(f"{description}: {ms} ms")

    return measure


def run_object_detection(frame_data: np.ndarray, timestamp: int):
    """
    Symbolic AI function. It gets the raw frame data, processes it,
    and returns its result along with its unique process ID.
    """

    from .ai import detect_objects

    worker_pid = mp.current_process().pid
    result = detect_objects(frame_data)
    return worker_pid, timestamp, frame_data, result


def on_done(future: Future):
    global max_output_timestamp
    active_futures.remove(future)

    try:
        worker_pid, timestamp, frame, detected_objects = future.result()
        if timestamp >= max_output_timestamp or True:
            max_output_timestamp = timestamp

            inference_time = detected_objects[0]
            # print("inference: ", inference_time)
            # print("update dashboard: ", inference_time)
            dashboard.update(worker_id=worker_pid, inference_time=inference_time)
            if live_stream_enabled.is_set():
                output_buffer.append((worker_pid, timestamp, frame, detected_objects[1]))
    except BrokenProcessPool as ex:
        print("Pool already broken, when future was done. Shutting down...")
    except KeyboardInterrupt:
        print("Future done, shutting down...")


# This class instance will hold the camera frames
class StreamingOutput(io.BufferedIOBase):
    def __init__(self, highlight_movement: bool = False) -> None:
        # self.frame: bytes = b""
        # self.condition = Condition()
        # self.backSub = cv2.createBackgroundSubtractorMOG2()
        print("StreamingOutput created")
        self.motion_detector = MotionDetector()
        self.highlight_movement = highlight_movement

    def write(self, buf: bytes) -> None:
        if self.closed:
            raise RuntimeError("Stream is closed")

        if len(active_futures) == NUM_AI_WORKERS:
            return

        # with self.condition:
        buf = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        buf: np.ndarray
            # self.frame = buf

        if buf is not None and buf.size:
            cv2.resize(buf, (640, 480), buf)
            has_movement = self.motion_detector.is_moving(buf)
            if has_movement and self.highlight_movement:
                measure_paint_movement = get_measure("Paint movement")
                highlighted_frame = self.motion_detector.highlight_movement_on(buf)
                buf = highlighted_frame  # todo: remove this after testing
                # measure_paint_movement()
            else:
                highlighted_frame = buf

            if has_movement and len(active_futures) < NUM_AI_WORKERS:
                try:
                    timestamp = time.monotonic_ns()

                    global max_output_timestamp

                    future: Future = process_pool.submit(
                        run_object_detection,
                        frame_data=buf[:],
                        timestamp=timestamp,
                    )

                    active_futures.append(future)
                    future.add_done_callback(on_done)

                except RuntimeError as e:
                    traceback.print_exc()
                    raise KeyboardInterrupt from e
            elif not has_movement and not active_futures:
                # print("not detecting on current frame")
                output_buffer.append((0, 0, buf, []))
            else:
                # this is for testing only
                output_buffer.append((0, 0, highlighted_frame, []))


class MotionDetector:
    def __init__(self, pixelcount_threshold: int = 500, denoise: bool = True):
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.foreground_mask: np.ndarray | None = None
        # self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.morph_kernel = np.ones((3,3), np.uint8)
        self.pixelcount_threshold = pixelcount_threshold
        self.denoise = denoise

    def moving_pixel_count(self):
        return cv2.countNonZero(self.foreground_mask)

    def is_moving(self, frame: np.ndarray):
        motion_ms = get_measure("Detect motion")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.denoise:
            cv2.GaussianBlur(frame, (33, 33), 0, frame)
        fg_mask = self.backSub.apply(frame)
        cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN, self.morph_kernel, fg_mask
        )
        self.foreground_mask = fg_mask
        is_moving = self.moving_pixel_count() >= self.pixelcount_threshold
        # motion_ms()
        return is_moving

    def get_boundinx_boxes(self, expansion_margin=25, size_threshold=200):
        """
        Merges nearby bounding boxes into a single one.

        Args:
            boxes (list): A list of bounding boxes, each in (x, y, w, h) format.
            expansion_margin (int): The margin in pixels to expand each box's check area.
                                    Boxes within this margin of each other will be merged.
            size_threshold:

        Returns:
            list: A new list of merged bounding boxes.
        """
        contours, _ = cv2.findContours(self.foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = [cv2.boundingRect(cnt) for cnt in contours]

        if not boxes:
            return []

        # A flag to indicate if any merges happened in a pass
        merges_made = True
        while merges_made:
            merges_made = False
            merged_boxes = []
            # Keep track of boxes that have been merged into another
            used_indices = set()

            for i in range(len(boxes)):
                if i in used_indices:
                    continue

                # Start a new merged box with the current box
                current_box = list(boxes[i])

                # Check against all other boxes
                for j in range(i + 1, len(boxes)):
                    if j in used_indices:
                        continue

                    other_box = boxes[j]

                    # Check for proximity. Expand current_box by the margin for the check.
                    # If the expanded current_box intersects with other_box, they are "close".
                    if (current_box[0] - expansion_margin < other_box[0] + other_box[2] and
                            current_box[0] + current_box[2] + expansion_margin > other_box[0] and
                            current_box[1] - expansion_margin < other_box[1] + other_box[3] and
                            current_box[1] + current_box[3] + expansion_margin > other_box[1]):

                        # We have a merge! Update the current merged box
                        x1 = min(current_box[0], other_box[0])
                        y1 = min(current_box[1], other_box[1])
                        x2 = max(current_box[0] + current_box[2], other_box[0] + other_box[2])
                        y2 = max(current_box[1] + current_box[3], other_box[1] + other_box[3])
                        current_box = [x1, y1, x2 - x1, y2 - y1]

                        # Mark the other box as used and flag that a merge happened
                        used_indices.add(j)
                        merges_made = True

                # Add the final state of the current box (either original or merged)
                merged_boxes.append(tuple(current_box))
                used_indices.add(i)

            # The list for the next pass is the result of the current pass's merges
            boxes = merged_boxes

        return [(x,y,w,h) for (x,y,w,h) in boxes if w*h >= size_threshold]

    def highlight_movement_on(
        self,
        frame: np.ndarray,
        transparency_factor: float = 0.4,
        overlay_color_bgr: tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        boxes = self.get_boundinx_boxes()
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        colored_overlay = np.full(frame.shape, overlay_color_bgr, dtype=np.uint8)
        blended = cv2.addWeighted(
            frame,
            1 - transparency_factor,
            colored_overlay,
            transparency_factor,
            0,
        )
        return np.where(
            self.foreground_mask[:, :, None] != 0,
            blended,
            frame,
        )


def _stream_cam_or_file_to(stream_output: StreamingOutput):


    resolution = (640, 480)

    try:
        from picamera2 import Picamera2, Preview
        from picamera2.encoders import JpegEncoder, MJPEGEncoder
        from picamera2.outputs import FileOutput, CircularOutput
        from libcamera import controls

        def stream_camera():
            try:
                print("Starting stream in 5 seconds...")
                time.sleep(5)
                picam2 = Picamera2()
                camera_config = picam2.create_video_configuration(
                    # main={"size": (1296, 972)},
                    main={"size": resolution},
                    # queue=False,
                    controls={
                        "FrameRate": 20,
                        "ColourGains": (1, 1),
                        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast,
                    },
                )
                picam2.configure(camera_config)
                picam2.start_preview(Preview.NULL)

                class MeasuringJpegEncoder(JpegEncoder):
                    def encode_func(self, request, name):
                        measure_encode = get_measure("JpegEncoder")
                        res = super().encode_func(request, name)
                        measure_encode()
                        return res

                class MeasuringMJPEGEncoder(MJPEGEncoder):
                    def encode(self, request, name):
                        measure_encode = get_measure("MJPEGEncoder")
                        res = super().encode(request, name)
                        measure_encode()
                        return res

                # encoder = MeasuringJpegEncoder(num_threads=1)
                # encoder = MeasuringMJPEGEncoder()
                encoder = MJPEGEncoder()
                # encoder = H264Encoder(100_000, repeat=True)

                # encoder.output = CircularOutput(file=stream_output, buffersize=10)
                # encoder.frame_skip_count = 10
                # encoder.use_hw = True

                picam2.start_recording(encoder, FileOutput(stream_output))
                # picam2.start()
                # if not encoder.running:
                #     picam2.start_encoder(encoder)
            except:
                traceback.print_exc()
                raise

            def cleanup():
                print("Cleaning up picam2")
                picam2.stop_encoder()
                picam2.stop_recording()
                encoder.stop()
                picam2.close()

            atexit.register(cleanup)

        streamer_func = stream_camera

    except ModuleNotFoundError as e:
        print(
            "Hardware initialization failed, using local file to stream",
            traceback.format_exc(),
        )
        # traceback.print_exc()

        # video_path = "/home/martin/Downloads/853889-hd_1280_720_25fps.mp4"
        # video_path = "/home/martin/Downloads/4039116-uhd_3840_2160_30fps.mp4"
        # video_path = "/home/martin/Downloads/cat.mov"
        video_path = "/home/martin/Downloads/VID_20250731_093415.mp4"
        video_path = "/mnt/c/Users/mbo20/Downloads/16701023-hd_1920_1080_60fps.mp4"
        # video_path = "/mnt/c/Users/mbo20/Downloads/20522838-hd_1080_1920_30fps.mp4"
        # video_path = "/home/martin/Downloads/output_converted.mov"
        # video_path = "/home/martin/Downloads/output_file.mov"
        # video_path = "/home/martin/Downloads/gettyimages-1382583689-640_adpp.mp4"

        def stream_file():
            if not os.path.isfile(video_path):
                raise ValueError("File not found: ", video_path)

            print("Starting videostream in 3s...")
            # this sleep is absolutely crucial!!! if we have a low-res video, opencv would be ready before django is and that breaks everything
            time.sleep(3)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Error: Could not open video at {video_path}")

            def close_video():
                cap.release()

            atexit.register(close_video)

            # max_fps = 30
            # motion_detector = MotionDetector()

            while True:
                if stream_output.closed:
                    break

                time.sleep(0.015)
                try:
                    ret, frame = cap.read()

                    if not ret:
                        # If it ended, rewind to the first frame (frame 0)
                        print("Video stream ended. Rewinding.")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue  # Skip the rest of this iteration and try reading the new first frame
                    else:
                        frame_resized = cv2.resize(frame, resolution)
                        frame_bytes = cv2.imencode(".jpeg", frame_resized)[1].tobytes()
                        stream_output.write(frame_bytes)
                except KeyboardInterrupt:
                    print("shutting down here!!!")
                    break
                except Exception:
                    traceback.print_exc()
                    raise
            print("Stopping filestream...")
            cap.release()

        if not os.path.isfile(video_path):
            raise RuntimeError(f"File not found: {video_path}")

        streamer_func = stream_file

    thread_pool.submit(streamer_func)


input_buffer = StreamingOutput(highlight_movement=True)
# This single instance will be imported by other parts of the app


def start_stream_nonblocking():
    _stream_cam_or_file_to(input_buffer)


def cleanup():
    print("[DJANGO SHUTDOWN] Stopping processes...")

    # with input_buffer.condition:
    input_buffer.close()

    thread_pool.shutdown(wait=True, cancel_futures=True)
    process_pool.shutdown(wait=True, cancel_futures=True)

    print("[DJANGO SHUTDOWN] Processes stopped.")


atexit.register(cleanup)
