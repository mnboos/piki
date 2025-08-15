import numpy as np
import cv2


def expand_roi_to_min_size(
    *, min_roi_size: int, roi: tuple[int, int, int, int], img_shape: tuple[int, int]
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
    target_size = max(min_roi_size, max(w, h))

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


def edge_distance(
    *, roi: tuple[int, int, int, int], img_shape: tuple[int, int]
) -> float:
    """Calculate distance to nearest edge for prioritization."""
    x, y, w, h = roi
    img_h, img_w = img_shape
    return min(x, y, img_w - (x + w), img_h - (y + h))


def cluster_with_constraints(
    *, boxes: list, max_dimension: int, merge_threshold: int = 999999
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
            cluster_xw = max(boxes[k][0] + boxes[k][2] for k in current_cluster_indices)
            cluster_yh = max(boxes[k][1] + boxes[k][3] for k in current_cluster_indices)
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
            max(boxes[k][0] + boxes[k][2] for k in current_cluster_indices) - final_x
        )
        final_h = (
            max(boxes[k][1] + boxes[k][3] for k in current_cluster_indices) - final_y
        )

        final_rois.append((final_x, final_y, final_w, final_h))

    return final_rois


def apply_non_max_suppression(*, boxes, overlap_threshold=0.3):
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
        final_boxes = [boxes[i] for i in np.array(indices_to_keep).flatten()]
    else:
        # If NMS suppressed everything, return an empty list
        final_boxes = []

    return final_boxes


def get_padded_roi_images(
    *,
    frame: np.ndarray,
    rois,
    preview_downscale_factor,
    target_size=320,
    pad_color=(0, 0, 0),
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
        w_hires, h_hires = (
            min(frame_w - x_hires, w_hires),
            min(frame_h - y_hires, h_hires),
        )

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

    return final_roi_images
