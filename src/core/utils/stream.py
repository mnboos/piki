import dataclasses
import os
import random
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Event, Queue, shared_memory, Lock
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp
from ctypes import c_float, c_int
import io
import time
import atexit
from concurrent.futures import ProcessPoolExecutor, Future, ThreadPoolExecutor
import cv2
import numpy as np
import traceback
from .metrics import LiveMetricsDashboard, retrieve_queue
from .helpers import MultiprocessingDequeue

NUM_AI_WORKERS: int = 1

active_futures: list[Future] = []
live_stream_enabled = Event()

dashboard_queue = retrieve_queue()
dashboard = LiveMetricsDashboard(retrieve_queue())

preview_downscale_factor = 2
ai_input_size = 320

max_output_timestamp = 0
output_buffer = MultiprocessingDequeue(queue=Queue(maxsize=10))

is_mask_streaming_enabled = Event()
is_object_detection_disabled = Event()
mask_output_buffer = MultiprocessingDequeue[np.ndarray](queue=Queue(maxsize=10))
mask_transparency = mp.Value(c_float, 0.5)

process_pool = ProcessPoolExecutor(max_workers=NUM_AI_WORKERS)
thread_pool = ThreadPoolExecutor(max_workers=1)

# --- New Globals for Shared Memory ---
SHM_NAME = "psm_frame_buffer"  # A unique name for our shared memory block
shm_lock = Lock()  # To synchronize access to the shared memory
shared_mem = None # Will hold the SharedMemory instance
shared_array = None # The numpy array view of the shared memory

@dataclasses.dataclass
class ForegroundMaskOptions:
    mog2_history = mp.Value(c_int, 500)
    mog2_var_threshold = mp.Value(c_int, 16)
    denoise_kernelsize = mp.Value(c_int, 7)


class TuningSettings:
    def __init__(self):
        self.foreground_mask_options = ForegroundMaskOptions()

    def update(self):
        pass


settings = TuningSettings()

def setup_shared_memory(frame_shape, frame_dtype):
    """Creates the shared memory block based on the first frame's properties."""
    global shared_mem, shared_array
    try:
        # Create a new shared memory block
        size = int(np.prod(frame_shape) * np.dtype(frame_dtype).itemsize)
        shared_mem = SharedMemory(create=True, size=size, name=SHM_NAME)
        print(f"Created shared memory block '{SHM_NAME}' with size {size / 1024**2:.2f} MB")
    except FileExistsError:
        # If it already exists from a previous crashed run, unlink it and retry
        print("Shared memory block already exists, unlinking and recreating.")
        SharedMemory(name=SHM_NAME).unlink()
        size = int(np.prod(frame_shape) * np.dtype(frame_dtype).itemsize)
        shared_mem = SharedMemory(create=True, size=size, name=SHM_NAME)

    # Create a NumPy array that uses the shared memory buffer
    shared_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shared_mem.buf)
    return shared_array

def cleanup_shared_memory():
    """Closes and unlinks the shared memory block on application exit."""
    global shared_mem
    print("Cleaning up shared memory...")
    if shared_mem:
        shared_mem.close()
        shared_mem.unlink() # Free the memory block

def get_measure(description: str):
    now = time.perf_counter()

    def measure():
        end = time.perf_counter()
        ms = str(round((end - now) * 1000, 2)).rjust(5)
        print(f"{description}: {ms} ms")

    return measure

def run_object_detection(
    shm_name: str,
    shape: tuple,
    dtype: np.dtype,
    rois,
    timestamp: int,
):
    existing_shm = None
    try:
        from .ai import detect_objects

        # --- Shared Memory Access ---
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        frame_in_shm = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        with shm_lock:
            frame_hires = frame_in_shm.copy()

        frame_h, frame_w, _ = frame_hires.shape
        worker_pid = mp.current_process().pid

        # --- AI Processing ---
        padded_images_and_details = MotionDetector.get_padded_roi_images(
            frame=frame_hires, rois=rois, target_size=ai_input_size
        )

        total_duration = 0
        all_detections = []

        for img, scale, effective_origin in padded_images_and_details:
            eff_orig_x, eff_orig_y = effective_origin

            duration, detections = detect_objects(img)
            total_duration += duration

            # 'detections' is a list of (label, confidence, local_pixel_bbox)
            # where local_pixel_bbox is (x, y, w, h)
            for label, confidence, local_pixel_bbox in detections:
                # --- START: Correct Transformation Logic ---

                # 1. Unpack the LOCAL PIXEL coordinates (x, y, w, h) from ai.py
                local_px_x, local_px_y, local_px_w, local_px_h = local_pixel_bbox

                # 2. Convert to (xmin, ymin, xmax, ymax) format
                local_px_xmin = local_px_x
                local_px_ymin = local_px_y
                local_px_xmax = local_px_x + local_px_w
                local_px_ymax = local_px_y + local_px_h

                # 3. Apply the 'scale' factor from the resizing step
                scaled_px_xmin = local_px_xmin * scale
                scaled_px_ymin = local_px_ymin * scale
                scaled_px_xmax = local_px_xmax * scale
                scaled_px_ymax = local_px_ymax * scale

                # 4. Translate by the 'effective_origin' to get GLOBAL PIXEL coordinates
                global_px_xmin = scaled_px_xmin + eff_orig_x
                global_px_ymin = scaled_px_ymin + eff_orig_y
                global_px_xmax = scaled_px_xmax + eff_orig_x
                global_px_ymax = scaled_px_ymax + eff_orig_y

                # 5. Re-normalize the FINAL GLOBAL pixel coordinates for views.py
                final_norm_coords = [
                    global_px_ymin / frame_h,
                    global_px_xmin / frame_w,
                    global_px_ymax / frame_h,
                    global_px_xmax / frame_w,
                ]

                all_detections.append((label, confidence, final_norm_coords))
                # --- END: Correct Transformation Logic ---

        result = total_duration, all_detections

        frame_lores = cv2.resize(
            frame_hires,
            None,
            fx=1 / preview_downscale_factor,
            fy=1 / preview_downscale_factor,
        )

        return worker_pid, timestamp, frame_lores, result

    except Exception as e:
        print(f"!!!!!!!!!!!!!! FATAL ERROR IN AI WORKER (PID: {os.getpid()}) !!!!!!!!!!!!!!")
        traceback.print_exc()
        raise e
    finally:
        if existing_shm is not None:
            existing_shm.close()

# def run_object_detection(
#     shm_name: str,
#     shape: tuple,
#     dtype: np.dtype,
#     rois,
#     timestamp: int,
# ):
#     """
#     Symbolic AI function. It gets the raw frame data, processes it,
#     and returns its result along with its unique process ID.
#     """
#
#     from .ai import detect_objects
#
#     # Attach to the existing shared memory block
#     existing_shm = shared_memory.SharedMemory(name=shm_name)
#
#     # Create a numpy array view pointing to the shared memory
#     frame_in_shm = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
#
#     # Acquire the lock, quickly make a local copy of the frame, then release the lock.
#     # This minimizes the time the main process is blocked from writing the next frame.
#     with shm_lock:
#         frame_hires = frame_in_shm.copy()
#
#     # We are done with the shared memory block in this process, so we can close it.
#     existing_shm.close()
#
#     # frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
#
#     padded_images_with_roi = MotionDetector.get_padded_roi_images(
#         frame=frame_hires, rois=rois
#     )
#     # if bboxes:
#     # rois = merge_boxes_opencv(boxes=bboxes, frame_shape=frame_data.shape[:2])
#     # print("rois: ", len(rois))
#
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # todo: should we do this before ai processing?
#     # lab = cv2.cvtColor(frame_data, cv2.COLOR_BGR2LAB)
#     # lab[:,:,0] = clahe.apply(lab[:,:,0])
#     # frame_data = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
#
#     worker_pid = mp.current_process().pid
#
#     # result = detect_objects(frame_hires)
#
#     total_duration = 0
#     all_detections = []
#     for img, roi, frame_coords in padded_images_with_roi:
#         rx, ry = frame_coords
#         duration, detections = detect_objects(img)
#         total_duration += duration
#         for label, confidence, local_coords in detections:
#             dx, dy, dw, dh = local_coords
#             new_x = dx + rx
#             new_y = dy + ry
#             all_detections.append((label, confidence, (new_x, new_y, dw, dh)))
#
#     result = total_duration, all_detections
#
#     # frame_resized = cv2.resize(frame_data, dsize=None, fx=1 / preview_downscale_factor, fy=1 / preview_downscale_factor)
#     # cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR, frame_resized)
#
#     # det = MotionDetector(320, 400)
#     # highlighted_frame = det.highlight_movement_on(frame=frame_lores, mask=fg_mask)
#
#     frame_lores = cv2.resize(
#         frame_hires,
#         None,
#         fx=1 / preview_downscale_factor,
#         fy=1 / preview_downscale_factor,
#     )
#
#     return worker_pid, timestamp, frame_lores, result
#     # return worker_pid, timestamp, frame_data, result


def on_done(future: Future):
    global max_output_timestamp
    active_futures.remove(future)

    try:
        worker_pid, timestamp, frame, detected_objects = future.result()
        if timestamp >= max_output_timestamp or True:
            max_output_timestamp = timestamp

            if detected_objects:
                inference_time, detections = detected_objects
                dashboard.update(worker_id=worker_pid, inference_time=inference_time)
                if live_stream_enabled.is_set():
                    output_buffer.append(
                        (worker_pid, timestamp, frame, detections)
                    )
    except BrokenProcessPool:
        print("Pool already broken, when future was done. Shutting down...")
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Future done, shutting down...")
    except:
        traceback.print_exc()
        raise


class MotionDetector:
    def __init__(self, min_roi_size: int, max_roi_size: int):
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            history=settings.foreground_mask_options.mog2_history.value,
            varThreshold=settings.foreground_mask_options.mog2_var_threshold.value,
        )
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # self.morph_kernel = np.ones((3, 3), np.uint8)
        self.pixelcount_threshold = 500

        self.min_area = 500
        self.min_roi_size = min_roi_size
        self.max_roi_size = max_roi_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _expand_roi_to_min_size(
        self, roi: tuple[int, int, int, int], img_shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """
        Expands an ROI from its center to a minimum target size, but does not
        shift the ROI if it hits a boundary. Instead, the expansion is clipped
        by the image edges.

        Args:
            roi (tuple): The initial (x, y, w, h) bounding box.
            img_shape (tuple): The (height, width) of the image frame.

        Returns:
            tuple: The final (x, y, w, h) of the expanded and clipped ROI.
        """
        x, y, w, h = roi
        img_h, img_w = img_shape

        # 1. Determine the target size. This ensures the ROI becomes at least min_roi_size
        #    while attempting to make it square if the original box was not.
        target_size = max(self.min_roi_size, max(w, h))

        # 2. Calculate the total padding needed for width and height
        pad_w = max(0, target_size - w)
        pad_h = max(0, target_size - h)

        # 3. Calculate the ideal padding for each side (half of the total)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left  # Handles odd numbers correctly

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # 4. --- The Critical Step: Limit padding by available space ---
        # The actual padding is the smaller of the ideal padding or the space
        # between the box edge and the frame edge.
        actual_pad_left = min(pad_left, x)
        actual_pad_right = min(pad_right, img_w - (x + w))
        actual_pad_top = min(pad_top, y)
        actual_pad_bottom = min(pad_bottom, img_h - (y + h))

        # 5. Calculate the final ROI coordinates based on the actual, clipped padding
        final_x = x - actual_pad_left
        final_y = y - actual_pad_top
        final_w = w + actual_pad_left + actual_pad_right
        final_h = h + actual_pad_top + actual_pad_bottom

        return int(final_x), int(final_y), int(final_w), int(final_h)

    @staticmethod
    def _edge_distance(
        roi: tuple[int, int, int, int], img_shape: tuple[int, int]
    ) -> float:
        """Calculate distance to nearest edge for prioritization."""
        x, y, w, h = roi
        img_h, img_w = img_shape
        return min(x, y, img_w - (x + w), img_h - (y + h))

    def is_moving(self, frame: np.ndarray):
        # motion_ms = get_measure("Detect motion")
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        # lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        # frame = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = self.clahe.apply(gray)

        denoise_kernelsize = settings.foreground_mask_options.denoise_kernelsize.value
        if denoise_kernelsize >= 1:
            kernel = (denoise_kernelsize,) * 2
            cv2.GaussianBlur(frame, kernel, 0, frame)

        fg_mask = self.backSub.apply(frame)
        # Remove noise
        cv2.dilate(fg_mask, self.morph_kernel, iterations=1, dst=fg_mask)
        cv2.erode(fg_mask, self.morph_kernel, iterations=2, dst=fg_mask)
        cv2.dilate(fg_mask, self.morph_kernel, iterations=1, dst=fg_mask)
        # cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel, fg_mask)
        # # Connect nearby regions (cat body parts)
        # cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel, fg_mask)
        is_moving = cv2.countNonZero(fg_mask) >= self.pixelcount_threshold
        # motion_ms()
        return is_moving, fg_mask

    @staticmethod
    def _cluster_with_constraints(
        boxes: list, max_dimension: int, merge_threshold: int = 999999
    ) -> list:
        """
        Private helper to greedily cluster boxes, respecting max size and proximity.
        """
        if not boxes:
            return []

        num_boxes = len(boxes)
        visited = [False] * num_boxes

        final_rois = []

        for i in range(num_boxes):
            if visited[i]:
                continue

            current_cluster_indices = {i}
            visited[i] = True

            while True:
                valid_candidates = []

                # Calculate the current cluster's bounding box
                cluster_x = min(boxes[k][0] for k in current_cluster_indices)
                cluster_y = min(boxes[k][1] for k in current_cluster_indices)
                cluster_xw = max(
                    boxes[k][0] + boxes[k][2] for k in current_cluster_indices
                )
                cluster_yh = max(
                    boxes[k][1] + boxes[k][3] for k in current_cluster_indices
                )
                cluster_w = cluster_xw - cluster_x
                cluster_h = cluster_yh - cluster_y
                cluster_cx = cluster_x + cluster_w / 2
                cluster_cy = cluster_y + cluster_h / 2

                for j in range(num_boxes):
                    if visited[j]:
                        continue

                    jx, jy, jw, jh = boxes[j]
                    dist = np.sqrt(
                        (cluster_cx - (jx + jw / 2)) ** 2
                        + (cluster_cy - (jy + jh / 2)) ** 2
                    )

                    if dist > merge_threshold:
                        continue

                    # --- Constraint Check BEFORE adding to candidates ---
                    potential_x = min(cluster_x, jx)
                    potential_y = min(cluster_y, jy)
                    potential_w = max(cluster_xw, jx + jw) - potential_x
                    potential_h = max(cluster_yh, jy + jh) - potential_y

                    if potential_w <= max_dimension and potential_h <= max_dimension:
                        valid_candidates.append((dist, j))

                if not valid_candidates:
                    break

                valid_candidates.sort()
                best_neighbor_idx = valid_candidates[0][1]

                visited[best_neighbor_idx] = True
                current_cluster_indices.add(best_neighbor_idx)

            # Finalize the cluster's bounding box
            final_x = min(boxes[k][0] for k in current_cluster_indices)
            final_y = min(boxes[k][1] for k in current_cluster_indices)
            final_w = (
                max(boxes[k][0] + boxes[k][2] for k in current_cluster_indices)
                - final_x
            )
            final_h = (
                max(boxes[k][1] + boxes[k][3] for k in current_cluster_indices)
                - final_y
            )

            final_rois.append((final_x, final_y, final_w, final_h))

        return final_rois

    @staticmethod
    def apply_non_max_suppression(boxes, overlap_threshold=0.3):
        """
        Applies Non-Max Suppression to a list of bounding boxes to remove redundant,
        overlapping ROIs.

        Args:
            boxes (list): A list of (x, y, w, h) bounding box tuples.
            overlap_threshold (float): The Intersection over Union (IoU) threshold.
                                       Boxes that overlap by more than this will be suppressed.
                                       A lower value is more aggressive.

        Returns:
            list: A final, clean list of non-overlapping bounding boxes.
        """
        if not boxes:
            return []

        # Ensure boxes are in a standard list of lists format
        # The NMS function can be picky about this.
        bbox_list = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in boxes]

        # Calculate scores (area) for each box
        scores = [w * h for (x, y, w, h) in bbox_list]

        # --- THE FIX ---
        # Convert scores to a NumPy array of float32, which is what NMSBoxes expects.
        scores_np = np.array(scores, dtype=np.float32)

        # The function requires a score_threshold, which we can set to 0 to consider all boxes.
        # It returns the *indices* of the boxes to keep.
        indices_to_keep = cv2.dnn.NMSBoxes(
            bboxes=bbox_list,
            scores=scores_np,  # Pass the correctly typed array
            score_threshold=0,
            nms_threshold=overlap_threshold,
        )

        # If NMS returns indices, use them to build the final list of boxes
        if len(indices_to_keep) > 0:
            # The indices can be a nested list, so we flatten them
            final_boxes = [boxes[i] for i in indices_to_keep.flatten()]
        else:
            # If NMS suppressed everything, return an empty list
            final_boxes = []

        return final_boxes

    def create_rois(self, *, mask: np.ndarray) -> list:
        """
        Finds blobs with connectedComponents, clusters them with constraints,
        and finalizes them to meet min_size requirements.

        Args:
            mask (np.ndarray): The input binary mask.

        Returns:
            list: A list of the final, fully optimized ROIs.
        """
        # 1. Find all individual blobs in the mask (Fast)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv2.CV_32S
        )

        # 2. Filter small blobs and collect their initial bounding boxes
        initial_boxes = []

        # if num_labels > 1:
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                fullness = area / (w * h)
                initial_boxes.append((x, y, w, h))

        if not initial_boxes:
            return []

        # # 3. Cluster the initial boxes with full constraints (Intelligent)
        # clustered_rois = self._cluster_with_constraints(
        #     initial_boxes,
        #     max_dimension=self.max_roi_size,
        # )
        clustered_rois = initial_boxes

        # 4. Finalize ROIs to enforce minimum size and handle edge cases (Format for AI)
        final_rois = []
        for roi_box in clustered_rois:
            # Assumes self._expand_roi_to_min_size is your boundary-aware finalization function
            final_roi = self._expand_roi_to_min_size(roi_box, mask.shape)
            final_rois.append(final_roi)

        # Optional: Sort final ROIs
        final_rois.sort(key=lambda roi: self._edge_distance(roi, mask.shape))

        return final_rois

    def get_bounding_boxes(
        self,
        foreground_mask: np.ndarray,
    ):
        # measure = get_measure("ROI retrieval")
        # res = self.method1_connected_components(foreground_mask)
        # res = self.method1_with_clustering(foreground_mask)
        res = self.create_rois(mask=foreground_mask)
        res = self.apply_non_max_suppression(res)
        # res = self.method2_watershed_segmentation(foreground_mask)
        # res = self.method3_mean_shift_clustering(foreground_mask)
        # res = self.method4_adaptive_threshold_contours(foreground_mask)
        # measure()
        return res

    @staticmethod
    def get_padded_roi_images(
        *, frame: np.ndarray, rois, target_size=320, pad_color=(0, 0, 0)
    ):
        """
        Crops ROIs from a frame, preserves their aspect ratio by either center-cropping
        or padding, and resizes them to a square target size.

        Returns:
            list: A list of tuples, where each tuple contains:
                  (padded_image, scale_factor, effective_origin_xy)
        """
        final_roi_images = []
        frame_h, frame_w, _ = frame.shape


        for current_roi in rois:
            # 1. Get HI-RES coordinates for the initial crop
            x_hires, y_hires, w_hires, h_hires = (
                int(current_roi[0] * preview_downscale_factor),
                int(current_roi[1] * preview_downscale_factor),
                int(current_roi[2] * preview_downscale_factor),
                int(current_roi[3] * preview_downscale_factor),
            )

            # Clamp to frame boundaries
            x_hires, y_hires = max(0, x_hires), max(0, y_hires)
            w_hires, h_hires = min(frame_w - x_hires, w_hires), min(frame_h - y_hires, h_hires)

            roi_crop = frame[y_hires : y_hires + h_hires, x_hires : x_hires + w_hires]

            # 2. Handle the different size cases to preserve aspect ratio
            crop_h, crop_w, _ = roi_crop.shape
            scale = 1.0
            if crop_h > target_size or crop_w > target_size:
                # --- CASE 1: ROI is LARGER than target ---
                # Crop from the center to maintain aspect ratio before resizing.

                # Find the shorter side
                min_dim = min(crop_h, crop_w)

                # Calculate the scaling factor
                scale = min_dim / target_size

                # Calculate the starting coordinates for a centered square crop
                start_x = (crop_w - min_dim) // 2
                start_y = (crop_h - min_dim) // 2

                # Perform the square crop from the original ROI
                square_crop = roi_crop[
                    start_y : start_y + min_dim, start_x : start_x + min_dim
                ]

                # Update the effective origin to account for the crop
                effective_origin = (x_hires + start_x, y_hires + start_y)

                # Resize the aspect-ratio-correct square crop to the target size
                final_image = cv2.resize(
                    square_crop,
                    (target_size, target_size),
                    interpolation=cv2.INTER_AREA,
                )

            else:
                # --- CASE 2: ROI is SMALLER than or equal to target ---
                # Pad the image to make it square.
                delta_w = target_size - crop_w
                delta_h = target_size - crop_h
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)

                final_image = cv2.copyMakeBorder(
                    roi_crop,
                    top,
                    bottom,
                    left,
                    right,
                    cv2.BORDER_CONSTANT,
                    value=pad_color,
                )

                # The effective origin is offset by the negative padding
                effective_origin = (x_hires - left, y_hires - top)
                # Scale remains 1.0 because we did not resize the original content

            # Assert that the final image is the correct size
            assert final_image.shape[:2] == (target_size, target_size), (
                "Final image processing failed."
            )

            final_roi_images.append((final_image, scale, effective_origin))

        # print("roi images: ", final_roi_images)
        return final_roi_images

    # @staticmethod
    # def get_padded_roi_images(
    #     *, frame: np.ndarray, rois, target_size=320, pad_color=(0, 0, 0)
    # ):
    #     """
    #     Takes clustered ROIs, crops them, and then uses cv2.copyMakeBorder to pad
    #     them to the minimum required size. This is the simpler, more robust method.
    #
    #     Args:
    #         frame (np.array): The original, full-resolution video frame.
    #         rois (list): A list of (x, y, w, h) tuples from the clustering step.
    #         target_size (int): The required dimension (width and height) for the final ROI.
    #         pad_color (tuple): The BGR color for the padding.
    #
    #     Returns:
    #         list: A list of final ROI images (as NumPy arrays), all of size min_dimension x min_dimension.
    #     """
    #     final_roi_images = []
    #     frame_h, frame_w, _ = frame.shape
    #
    #     for current_roi in rois:
    #         x, y, w, h = current_roi
    #         # 1. Crop the ROI from the high-resolution frame first.
    #         # Ensure cropping coordinates are integers and within bounds.
    #         x, y, w, h = (
    #             max(0, x * preview_downscale_factor),
    #             max(0, y * preview_downscale_factor),
    #             int(w * preview_downscale_factor),
    #             int(h * preview_downscale_factor),
    #         )
    #
    #         w = min(frame_w - x, w)
    #         h = min(frame_h - y, h)
    #
    #         roi_crop = frame[y : y + h, x : x + w]
    #         # if the roi is bigger than the targetsize in any direction, crop and resize
    #         if (
    #             roi_crop.shape[0] > target_size
    #             or roi_crop.shape[1] > target_size
    #         ):
    #             min_crop_size = min(roi_crop.shape[0], roi_crop.shape[1])
    #             center = roi_crop.shape
    #             x = int(center[1] / 2 - min_crop_size / 2)
    #             y = int(center[0] / 2 - min_crop_size / 2)
    #             cropped = roi_crop[y : y + min_crop_size, x : x + min_crop_size]
    #             scale = min_crop_size / target_size
    #             x *= scale
    #             y *= scale
    #             padded_image = cv2.resize(cropped, (target_size, target_size))
    #
    #         elif roi_crop.shape[0] < target_size or roi_crop.shape[1] < target_size:
    #             # Get the current dimensions of the crop
    #             current_h, current_w, _ = roi_crop.shape
    #
    #             # 2. Calculate how much padding is needed for width and height
    #             delta_w = max(0, target_size - current_w)
    #             delta_h = max(0, target_size - current_h)
    #
    #             # 3. Distribute the padding to top/bottom and left/right
    #             # This ensures the original crop stays centered in the padded image.
    #             top = delta_h // 2
    #             bottom = delta_h - top  # Handles odd numbers correctly
    #
    #             left = delta_w // 2
    #             right = delta_w - left
    #
    #             # 4. Use cv2.copyMakeBorder to add the black padding
    #             padded_image = cv2.copyMakeBorder(
    #                 roi_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
    #             )
    #         else:
    #             padded_image = roi_crop
    #
    #         assert padded_image.shape[0]==target_size and padded_image.shape[1]==target_size, f"Padded image has wrong dimensions. Expected ({target_size}, {target_size}), actual {padded_image.shape[:2]}"
    #
    #         final_roi_images.append((padded_image, current_roi, (x,y)))
    #
    #     return final_roi_images

    def highlight_movement_on(
        self,
        *,
        frame: np.ndarray,
        mask: np.ndarray,
        transparency_factor: float = 0.4,
        overlay_color_rgb: tuple[int, int, int] = (255, 0, 0),
        draw_boxes: bool = True,
    ) -> np.ndarray:
        if draw_boxes:
            boxes = self.get_bounding_boxes(mask)
            for x, y, w, h in boxes:
                # rect_color = (0, 0, 255)
                rect_color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

        colored_overlay = np.full(
            frame.shape, overlay_color_rgb, dtype=np.uint8
        )  # todo: do this only once
        blended = cv2.addWeighted(
            frame,
            transparency_factor,
            colored_overlay,
            1 - transparency_factor,
            0,
        )
        return np.where(
            mask[:, :, None] != 0,
            blended,
            frame,
        )

class FrameHandler:
    def __init__(self, highlight_movement: bool = False):
        min_roi_size = int(ai_input_size / preview_downscale_factor)
        max_roi_size = int((ai_input_size + 100) / preview_downscale_factor)

        self.motion_detector = MotionDetector(
            min_roi_size=min_roi_size, max_roi_size=max_roi_size
        )
        self.highlight_movement = highlight_movement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def update_motion_detector(self):
        self.motion_detector = MotionDetector(
            min_roi_size=self.motion_detector.min_roi_size,
            max_roi_size=self.motion_detector.max_roi_size,
        )

    def process_frame(self, frame_hires: np.ndarray):
        global shared_array

        # --- Shared Memory Logic ---
        # On the first run, create the shared memory block
        if shared_array is None:
            setup_shared_memory(frame_hires.shape, frame_hires.dtype)

        # Lock the shared memory, copy the new frame data into it, and then unlock.
        # This ensures a worker process doesn't read a partially updated frame.
        with shm_lock:
            shared_array[:] = frame_hires

        frame_lores = cv2.resize(
            frame_hires,
            None,
            fx=1 / preview_downscale_factor,
            fy=1 / preview_downscale_factor,
        )

        # with self.condition:
        # buf = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        # buf: np.ndarray
        # self.frame = buf

        if frame_lores is not None and frame_lores.size:
            has_movement, mask = self.motion_detector.is_moving(frame_lores)
            if is_mask_streaming_enabled.is_set():
                # clahe_recolored = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
                buf_highlighted = self.motion_detector.highlight_movement_on(
                    frame=frame_lores,
                    mask=mask,
                    overlay_color_rgb=(
                        147,
                        20,
                        255,
                    ),
                    transparency_factor=mask_transparency.value,
                    draw_boxes=False,
                )
                mask_output_buffer.append(buf_highlighted)

            if has_movement and len(active_futures) < NUM_AI_WORKERS:
                try:
                    timestamp = time.monotonic_ns()

                    global max_output_timestamp

                    # buf_hires = cv2.imdecode(np.frombuffer(buf_hires, dtype=np.uint8), cv2.IMREAD_COLOR)
                    # buf_hires: np.ndarray

                    rois = self.motion_detector.create_rois(mask=mask)

                    # worker_pid, timestamp, frame, detected_objects = run_object_detection(
                    #     shm_name=SHM_NAME,
                    #     shape=frame_hires.shape,
                    #     dtype=frame_hires.dtype,
                    #     rois=rois,
                    #     timestamp=timestamp,)
                    # if timestamp >= max_output_timestamp or True:
                    #     max_output_timestamp = timestamp
                    #
                    #     if detected_objects:
                    #         inference_time, detections = detected_objects
                    #         dashboard.update(worker_id=worker_pid, inference_time=inference_time)
                    #         if live_stream_enabled.is_set():
                    #             output_buffer.append(
                    #                 (worker_pid, timestamp, frame, detections))

                    future: Future = process_pool.submit(
                        run_object_detection,
                        shm_name=SHM_NAME,
                        shape=frame_hires.shape,
                        dtype=frame_hires.dtype,
                        rois=rois,
                        timestamp=timestamp,
                    )
                    active_futures.append(future)
                    future.add_done_callback(on_done)

                except:
                    traceback.print_exc()
                    raise
            elif not has_movement and not active_futures:
                # print("not detecting on current frame")
                output_buffer.append((0, 0, frame_lores, []))
            else:
                # this is for testing only, this will actually rewind a little
                # output_buffer.append((0, 0, frame_lores, []))
                pass


# This class instance will hold the camera frames
class StreamingOutput(io.BufferedIOBase):
    def __init__(self, highlight_movement: bool = False) -> None:
        self.frame_handler = FrameHandler(highlight_movement=highlight_movement)

    def write(self, buf_hires: bytes) -> None:
        if self.closed:
            raise RuntimeError("Stream is closed")

        if len(active_futures) == NUM_AI_WORKERS:
            return

        buf_hires = cv2.imdecode(
            np.frombuffer(buf_hires, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        self.frame_handler.process_frame(frame_hires=buf_hires)


def _stream_cam_or_file_to(stream_output: StreamingOutput):
    try:
        from picamera2 import Picamera2, Preview
        from picamera2.encoders import JpegEncoder, MJPEGEncoder
        from picamera2.outputs import FileOutput
        from libcamera import controls

        def stream_camera():
            try:
                print("Starting stream in 5 seconds...")
                time.sleep(5)
                picam2 = Picamera2()
                # resolution = (640, 480)
                camera_config = picam2.create_video_configuration(
                    # main={"size": (1296, 972)},
                    # main={"size": resolution},
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


                encoder = MJPEGEncoder()

                picam2.start_recording(encoder, FileOutput(stream_output))
            except:
                traceback.print_exc()
                raise

            def cleanup_camera():
                print("Cleaning up picam2")
                picam2.stop_encoder()
                picam2.stop_recording()
                encoder.stop()
                picam2.close()

            atexit.register(cleanup_camera)

        if os.environ.get("MOCK_CAMERA"):
            streamer_func = get_file_streamer(stream_output)
        else:
            streamer_func = stream_camera

    except ModuleNotFoundError:
        print(
            "Hardware initialization failed, using local file to stream",
            traceback.format_exc(),
        )
        streamer_func = get_file_streamer(stream_output)

    thread_pool.submit(streamer_func)


def get_file_streamer(stream_output: StreamingOutput):
    video_path = os.environ.get("MOCK_CAMERA_VIDEO_PATH")
    if not video_path:
        video_path = "/home/martin/Downloads/853889-hd_1280_720_25fps.mp4"
        # video_path = "/home/martin/Downloads/4039116-uhd_3840_2160_30fps.mp4"
        # video_path = "/home/martin/Downloads/cat.mov"
        # video_path = "/home/martin/Downloads/VID_20250731_093415.mp4"
        video_path = "/mnt/c/Users/mbo20/Downloads/16701023-hd_1920_1080_60fps.mp4"
        # video_path = "/mnt/c/Users/mbo20/Downloads/20522838-hd_1080_1920_30fps.mp4"
        # video_path = "/mnt/c/Users/mbo20/Downloads/13867923-uhd_3840_2160_30fps.mp4"
        # video_path = "/home/martin/Downloads/VID_20250808_080717.mp4"
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

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        sleep_time = round(1 / fps, 4)
        print("FPS: ", fps, sleep_time)

        def close_video():
            cap.release()

        atexit.register(close_video)

        frame_handler = FrameHandler(highlight_movement=True)

        last_time = time.perf_counter()

        while True:
            if stream_output.closed:
                break

            now = time.perf_counter()
            time_passed = now - last_time
            last_time = now

            remaining_sleep_time = max(sleep_time - time_passed, 0.05)
            time.sleep(remaining_sleep_time)
            try:
                ret, frame = cap.read()

                if not ret:
                    # If it ended, rewind to the first frame (frame 0)
                    print("Video stream ended. Rewinding.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue  # Skip the rest of this iteration and try reading the new first frame
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame_bytes_hires = cv2.imencode(".jpeg", frame)[1].tobytes()
                    # stream_output.write(frame_bytes_hires)
                    frame_handler.process_frame(frame_hires=frame)
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
    return stream_file


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
    cleanup_shared_memory()

    print("[DJANGO SHUTDOWN] Processes stopped.")

atexit.register(cleanup)