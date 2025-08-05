import dataclasses
import os
import random
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


def get_measure(description: str):
    now = time.perf_counter()

    def measure():
        end = time.perf_counter()
        ms = str(round((end - now) * 1000, 2)).rjust(5)
        print(f"{description}: {ms} ms")

    return measure


def finalize_rois_boundary_aware(*, frame_shape, clustered_rois, min_dimension=300):
    """
    Takes clustered ROIs and ensures each one meets the minimum dimension.
    This function intelligently shifts the ROI when expanding near the frame edges
    to satisfy the size constraint wherever physically possible.

    Args:
        frame_shape (tuple): The (height, width) of the original frame.
        clustered_rois (list): A list of (x, y, w, h) tuples from the clustering step.
        min_dimension (int): The required minimum width AND height for a final ROI.

    Returns:
        list: The final list of ROI coordinates, ready for cropping.
    """
    final_rois = []
    frame_h, frame_w = frame_shape

    for x, y, w, h in clustered_rois:
        # Determine the final target size, ensuring it meets the minimum
        final_w = max(w, min_dimension)
        final_h = max(h, min_dimension)

        # --- Intelligent Positioning Logic ---

        # Start by trying to center the new, larger box over the original box's center
        center_x = x + w // 2
        center_y = y + h // 2

        new_x = center_x - (final_w // 2)
        new_y = center_y - (final_h // 2)

        # --- Boundary Correction: Shift the box if it goes out of bounds ---

        # Check left edge
        if new_x < 0:
            new_x = 0
        # Check top edge
        if new_y < 0:
            new_y = 0

        # Check right edge
        if new_x + final_w > frame_w:
            new_x = frame_w - final_w
        # Check bottom edge
        if new_y + final_h > frame_h:
            new_y = frame_h - final_h

        # The final dimensions are clamped to the frame size as a last resort
        # This only matters if min_dimension is larger than the frame itself.
        final_w = min(final_w, frame_w)
        final_h = min(final_h, frame_h)

        final_rois.append((int(new_x), int(new_y), int(final_w), int(final_h)))

    return final_rois


def constrained_agglomerative_clustering(
    *, boxes, max_dimension=400, merge_threshold=640
):
    """
    Iteratively finds the closest pair of bounding box clusters and merges them,
    respecting a maximum size constraint.

    Args:
        boxes (list): A list of initial (x, y, w, h) bounding boxes.
        max_dimension (int): The maximum allowed width or height for a merged ROI.
        merge_threshold (int): The max distance between cluster centers to consider a merge.

    Returns:
        list: The final list of merged and optimized ROIs.
    """
    if not boxes:
        return []

    # Start with each box being its own cluster
    clusters = [[box] for box in boxes]

    while True:
        min_dist = float("inf")
        best_pair = None

        # --- Find the globally best pair to merge ---
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Get the overall bounding box of each cluster
                c1_boxes = clusters[i]
                c2_boxes = clusters[j]

                c1_x = min(b[0] for b in c1_boxes)
                c1_y = min(b[1] for b in c1_boxes)
                c1_w = max(b[0] + b[2] for b in c1_boxes) - c1_x
                c1_h = max(b[1] + b[3] for b in c1_boxes) - c1_y

                c2_x = min(b[0] for b in c2_boxes)
                c2_y = min(b[1] for b in c2_boxes)
                c2_w = max(b[0] + b[2] for b in c2_boxes) - c2_x
                c2_h = max(b[1] + b[3] for b in c2_boxes) - c2_y

                # Calculate distance between cluster centers
                dist = np.sqrt(
                    ((c1_x + c1_w / 2) - (c2_x + c2_w / 2)) ** 2
                    + ((c1_y + c1_h / 2) - (c2_y + c2_h / 2)) ** 2
                )

                if dist < min_dist:
                    min_dist = dist
                    best_pair = (i, j)

        # --- Check if a valid merge was found ---
        if best_pair is not None and min_dist < merge_threshold:
            i, j = best_pair

            # --- Validate the potential merge against constraints ---
            merged_cluster_boxes = clusters[i] + clusters[j]

            merged_x = min(b[0] for b in merged_cluster_boxes)
            merged_y = min(b[1] for b in merged_cluster_boxes)
            merged_w = max(b[0] + b[2] for b in merged_cluster_boxes) - merged_x
            merged_h = max(b[1] + b[3] for b in merged_cluster_boxes) - merged_y

            if merged_w <= max_dimension and merged_h <= max_dimension:
                # If valid, perform the merge
                # Create a new list of clusters, replacing the merged pair with the new cluster
                new_clusters = []
                for k in range(len(clusters)):
                    if k != i and k != j:
                        new_clusters.append(clusters[k])
                new_clusters.append(merged_cluster_boxes)
                clusters = new_clusters
            else:
                # If the best pair can't be merged, no smaller-distance pairs can either.
                # We can stop searching for this pass. A more advanced implementation
                # could continue searching for other valid pairs, but for simplicity,
                # we stop when the absolute best option is invalid.
                break
        else:
            # If no pairs are close enough to merge, we are done.
            break

    # --- Final Step: Convert the final clusters to bounding boxes ---
    final_rois = []
    for cluster in clusters:
        x = min(b[0] for b in cluster)
        y = min(b[1] for b in cluster)
        w = max(b[0] + b[2] for b in cluster) - x
        h = max(b[1] + b[3] for b in cluster) - y
        final_rois.append((x, y, w, h))

    return final_rois


def cluster_with_size_limit(*, boxes, max_dimension=400, margin=50):
    """
    Greedily clusters boxes, ensuring no single cluster's bounding box
    exceeds a maximum dimension.

    Args:
        boxes (list): A list of initial (x, y, w, h) bounding boxes.
        max_dimension (int): The maximum allowed width or height for a final ROI.
        margin (int): The pixel distance to consider boxes "neighbors."

    Returns:
        list: A list of the final, size-constrained, merged ROIs.
    """
    if not boxes:
        return []

    num_boxes = len(boxes)
    # Keep track of which boxes have been assigned to a cluster
    visited = [False] * num_boxes

    final_rois = []

    for i in range(num_boxes):
        if visited[i]:
            continue  # Skip boxes that are already in a cluster

        # This box is the seed for a new cluster
        current_cluster_boxes = [boxes[i]]
        visited[i] = True

        # This is the bounding box of the entire cluster so far
        cx, cy, cw, ch = boxes[i]

        # --- Iteratively grow the current cluster ---
        while True:
            found_a_merge_in_this_pass = False
            for j in range(num_boxes):
                if visited[j]:
                    continue

                # Check if box j is a neighbor of the current cluster's bounding box
                jx, jy, jw, jh = boxes[j]

                # The "closeness" check with margin
                if (
                    cx < jx + jw + margin
                    and cx + cw + margin > jx
                    and cy < jy + jh + margin
                    and cy + ch + margin > jy
                ):
                    # --- Pre-calculate the potential new bounding box ---
                    potential_x = min(cx, jx)
                    potential_y = min(cy, jy)
                    potential_xw = max(cx + cw, jx + jw)
                    potential_yh = max(cy + ch, jy + jh)
                    potential_w = potential_xw - potential_x
                    potential_h = potential_yh - potential_y

                    # --- CRITICAL: Check the size constraint BEFORE merging ---
                    if potential_w > max_dimension or potential_h > max_dimension:
                        # This merge would make the ROI too big, so we skip it
                        continue

                    # If the size is okay, perform the merge
                    visited[j] = True
                    # Update the cluster's overall bounding box
                    cx, cy, cw, ch = potential_x, potential_y, potential_w, potential_h
                    found_a_merge_in_this_pass = True

            # If we went through a whole pass without finding a valid merge,
            # this cluster is finished growing.
            if not found_a_merge_in_this_pass:
                break

        # The cluster is final, add its bounding box to the results
        final_rois.append((cx, cy, cw, ch))

    return final_rois


def cluster_boxes_opencv(*, boxes, frame_shape, eps=150):
    """
    Clusters bounding boxes based on proximity using only OpenCV's image operations.
    This is a visual equivalent to DBSCAN for this specific problem.

    Args:
        boxes (list): A list of (x, y, w, h) bounding box tuples.
        frame_shape (tuple): The (height, width) of the original frame.
        eps (int): The "merge radius" in pixels. Boxes within this distance of each
                   other will be grouped. This is the most important tuning parameter.

    Returns:
        list: A list of new (x, y, w, h) tuples representing the merged ROI for each cluster.
    """
    if not boxes:
        return []

    # 1. Create a blank canvas
    canvas = np.zeros(frame_shape, dtype=np.uint8)

    # 2. Draw each bounding box as a filled, white rectangle
    for x, y, w, h in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), 255, -1)  # -1 fills the rectangle

    # 3. Dilate to connect nearby blobs. The kernel size is our 'eps'
    # The kernel must have an odd size.
    kernel_size = int(eps)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure it's odd

    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_canvas = cv2.dilate(canvas, kernel, iterations=1)
    else:
        dilated_canvas = canvas  # No dilation if eps is too small

    # 4. Find the contours of the final merged blobs
    contours, _ = cv2.findContours(
        dilated_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 5. Get the bounding box for each final contour
    merged_rois = []
    for cnt in contours:
        merged_rois.append(cv2.boundingRect(cnt))

    return merged_rois


def run_object_detection(frame_data: np.ndarray, fg_mask: np.ndarray, timestamp: int):
    """
    Symbolic AI function. It gets the raw frame data, processes it,
    and returns its result along with its unique process ID.
    """

    from .ai import detect_objects

    bboxes = MotionDetector.get_bounding_boxes(fg_mask)
    # if bboxes:
    # rois = merge_boxes_opencv(boxes=bboxes, frame_shape=frame_data.shape[:2])
    # print("rois: ", len(rois))

    worker_pid = mp.current_process().pid
    if bboxes:
        result = detect_objects(frame_data)
    else:
        result = 0, []

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
                output_buffer.append(
                    (worker_pid, timestamp, frame, detected_objects[1])
                )
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
            has_movement, mask = self.motion_detector.is_moving(buf)
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
                        fg_mask=mask,
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
        self.morph_kernel = np.ones((3, 3), np.uint8)
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
        cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel, fg_mask)
        self.foreground_mask = fg_mask
        is_moving = self.moving_pixel_count() >= self.pixelcount_threshold
        # motion_ms()
        return is_moving, fg_mask

    @staticmethod
    def get_bounding_boxes(
        foreground_mask: np.ndarray, expansion_margin=25, size_threshold=200
    ):
        """
        Merges nearby bounding boxes into a single one.

        Args:
            foreground_mask (list): .
            expansion_margin (int): The margin in pixels to expand each box's check area.
                                    Boxes within this margin of each other will be merged.
            size_threshold:

        Returns:
            list: A new list of merged bounding boxes.
        """
        contours, _ = cv2.findContours(
            foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = [cv2.boundingRect(cnt) for cnt in contours]

        merges_made = len(boxes) > 1
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
                    if (
                        current_box[0] - expansion_margin < other_box[0] + other_box[2]
                        and current_box[0] + current_box[2] + expansion_margin
                        > other_box[0]
                        and current_box[1] - expansion_margin
                        < other_box[1] + other_box[3]
                        and current_box[1] + current_box[3] + expansion_margin
                        > other_box[1]
                    ):
                        # We have a merge! Update the current merged box
                        x1 = min(current_box[0], other_box[0])
                        y1 = min(current_box[1], other_box[1])
                        x2 = max(
                            current_box[0] + current_box[2], other_box[0] + other_box[2]
                        )
                        y2 = max(
                            current_box[1] + current_box[3], other_box[1] + other_box[3]
                        )
                        current_box = [x1, y1, x2 - x1, y2 - y1]

                        # Mark the other box as used and flag that a merge happened
                        used_indices.add(j)
                        merges_made = True

                # Add the final state of the current box (either original or merged)
                merged_boxes.append(tuple(current_box))
                used_indices.add(i)

            # The list for the next pass is the result of the current pass's merges
            boxes = merged_boxes

        # boxes = [(x, y, w, h) for (x, y, w, h) in boxes if w * h >= size_threshold]

        high_res_h, high_res_w = (1080, 1920)
        low_res_h, low_res_w = foreground_mask.shape[:2]
        scale_x = high_res_w / low_res_w
        scale_y = high_res_h / low_res_h

        def scale_boxes(*, boxes_to_scale, factor_x, factor_y):
            return [
                (
                    int(x * factor_x),
                    int(y * factor_y),
                    int(w * factor_x),
                    int(h * factor_y),
                )
                for (x, y, w, h) in boxes_to_scale
            ]

        high_res_rois = scale_boxes(
            boxes_to_scale=boxes, factor_x=scale_x, factor_y=scale_y
        )

        rois_already_too_big = []
        rois_to_cluster = []

        ROI_MAX_SIZE = 300

        for x, y, w, h in high_res_rois:
            if w > ROI_MAX_SIZE or h > ROI_MAX_SIZE:
                # This box is already too big, so it bypasses clustering.
                rois_already_too_big.append((x, y, w, h))
            else:
                # This box is a candidate for merging.
                rois_to_cluster.append((x, y, w, h))

        clustered_high_res_rois = constrained_agglomerative_clustering(
            boxes=rois_to_cluster
        )

        # high_res_h, high_res_w = high_res_frame.shape[:2]
        # low_res_h, low_res_w = low_res_frame.shape[:2]

        # for x, y, w, h in clustered_low_res_rois:
        #     scaled_x = x * scale_x
        #     scaled_y = y * scale_y
        #     scaled_w = w * scale_x
        #     scaled_h = h * scale_y
        #     scaled_rois.append((scaled_x, scaled_y, scaled_w, scaled_h))

        finalized_highres_boxes = finalize_rois_boundary_aware(
            frame_shape=(high_res_h, high_res_w),
            clustered_rois=clustered_high_res_rois + rois_already_too_big,
            min_dimension=300,
        )
        # for box in finalized_highres_boxes:
        #     x, y, w, h = box
        #     print("box: ", w, h)

        downscaled_rois = scale_boxes(
            boxes_to_scale=finalized_highres_boxes,
            factor_x=1 / scale_x,
            factor_y=1 / scale_y,
        )
        # for x, y, w, h in finalized_highres_boxes:
        #     scaled_x = int(x / scale_x)
        #     scaled_y = int(y / scale_y)
        #     scaled_w = int(w / scale_x)
        #     scaled_h = int(h / scale_y)
        #     downscaled_rois.append((scaled_x, scaled_y, scaled_w, scaled_h))

        return downscaled_rois
        # return cluster_boxes_opencv(boxes=boxes, frame_shape=foreground_mask.shape[:2])

    def highlight_movement_on(
        self,
        frame: np.ndarray,
        transparency_factor: float = 0.4,
        overlay_color_bgr: tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        boxes = self.get_bounding_boxes(self.foreground_mask)
        for x, y, w, h in boxes:
            # rect_color = (0, 0, 255)
            rect_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

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
        video_path = "/home/martin/Downloads/4039116-uhd_3840_2160_30fps.mp4"
        # video_path = "/home/martin/Downloads/cat.mov"
        video_path = "/home/martin/Downloads/VID_20250731_093415.mp4"
        # video_path = "/mnt/c/Users/mbo20/Downloads/16701023-hd_1920_1080_60fps.mp4"
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
