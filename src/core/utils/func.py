import cv2
import numpy as np

from .interfaces import Box


def expand_roi_to_min_size(
    *,
    min_roi_size: int,
    roi: tuple[int, int, int, int],
    img_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Expand a ROI from its center to a minimum target size.

     Does not shift the ROI if it hits a boundary. Instead, the expansion is clipped by the image edges.

    Args:
        min_roi_size (int): The minmimal size of the roi.
        roi (tuple): The initial (x, y, w, h) bounding box.
        img_shape (tuple): The (height, width) of the image frame.

    Returns:
        tuple: The final (x, y, w, h) of the expanded and clipped ROI.

    """
    x, y, w, h = roi
    img_h, img_w = img_shape

    # 1. Determine the target size. This ensures the ROI becomes at least min_roi_size
    #    while attempting to make it square if the original box was not.
    target_size = max(min_roi_size, w, h)

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
    *,
    roi: tuple[int, int, int, int],
    img_shape: tuple[int, int],
) -> float:
    """Calculate distance to nearest edge for prioritization."""
    x, y, w, h = roi
    img_h, img_w = img_shape
    return min(x, y, img_w - (x + w), img_h - (y + h))


def cluster_with_constraints(
    *,
    boxes: list,
    max_dimension: int,
    merge_threshold: int = 999999,
) -> list:
    """Private helper to greedily cluster boxes, respecting max size and proximity."""
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
                    (cluster_cx - (jx + jw / 2)) ** 2 + (cluster_cy - (jy + jh / 2)) ** 2,
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
        final_w = max(boxes[k][0] + boxes[k][2] for k in current_cluster_indices) - final_x
        final_h = max(boxes[k][1] + boxes[k][3] for k in current_cluster_indices) - final_y

        final_rois.append((final_x, final_y, final_w, final_h))

    return final_rois


def apply_non_max_suppression(*, boxes: list[Box], overlap_threshold: float = 0.3):
    """Apply Non-Max Suppression to a list of bounding boxes to remove redundant, overlapping ROIs.

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
    # bbox_list = [Box(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]

    # Calculate scores (area) for each box
    scores = [w * h for (x, y, w, h) in boxes]

    # --- THE FIX ---
    # Convert scores to a NumPy array of float32, which is what NMSBoxes expects.
    scores_np = np.array(scores, dtype=np.float32)

    # The function requires a score_threshold, which we can set to 0 to consider all boxes.
    # It returns the *indices* of the boxes to keep.
    indices_to_keep = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores_np,  # Pass the correctly typed array
        score_threshold=0,
        nms_threshold=overlap_threshold,
    )

    if len(indices_to_keep):
        return [boxes[i] for i in np.array(indices_to_keep).flatten()]
    return []


def _slice_bgr_tile(frame: np.ndarray, tx: int, ty: int, tile_size: int) -> np.ndarray:
    return frame[ty: ty + tile_size, tx: tx + tile_size].copy()


def _slice_nv12_tile(nv12: np.ndarray, frame_w: int, frame_h: int,
                     tx: int, ty: int, tile_size: int) -> np.ndarray:
    """Slice a tile from a flat NV12 array without any colorspace conversion.

    NV12 layout: Y plane (frame_h rows) followed by interleaved UV plane (frame_h/2 rows).
    Chroma is 4:2:0 so UV coords are halved.
    """
    y_plane = nv12[:frame_h]
    uv_plane = nv12[frame_h:]
    y_tile = y_plane[ty: ty + tile_size, tx: tx + tile_size]
    uv_tile = uv_plane[ty // 2: (ty + tile_size) // 2, tx: tx + tile_size]
    return np.vstack([y_tile, uv_tile])


def get_stereo_stripe_tiles(
    frame: np.ndarray,
    tile_size: int = 640,
    active_width: int = 1280,  # 640 (Left) + 640 (Right)
    active_height: int = 352,
    is_nv12: bool = True,
) -> list[tuple[np.ndarray, int, int]]:
    """
    Slices the wide stereo image into horizontal 640x640 tiles.
    Focuses on the vertical 'middle stripe' of the active image content.
    """
    buffer_w = frame.shape[1]  # 1920
    buffer_h = (frame.shape[0] * 2 // 3) if is_nv12 else frame.shape[0]  # 1080

    tiles = []

    # Calculate Vertical Offset to center the 352px image in the 640px model tile
    # ty = 0 if you want it top-aligned, but centering is better for many models.
    # However, since stereonet writes to the top-left (0,0), we'll use ty=0
    # to ensure we actually catch all the data.
    ty = 0

    # We iterate horizontally across the active width (Left then Right)
    # 0 -> 640 (Tile 1: Left Camera)
    # 640 -> 1280 (Tile 2: Right Camera)
    for tx in range(0, active_width, tile_size):
        if tx + tile_size > buffer_w:
            break

        if is_nv12:
            # Zero-copy NV12 slice
            tile = _slice_nv12_tile(frame, buffer_w, buffer_h, tx, ty, tile_size)
        else:
            # Zero-copy BGR slice
            tile = frame[ty : ty + tile_size, tx : tx + tile_size]

        tiles.append((tile, tx, ty))

    return tiles


def slice_roi_into_tiles(
    *,
    frame: np.ndarray,
    rois: list[Box],
    tile_size: int,
    preview_downscale_factor: float,
    model_input_type: str = "BGR",
) -> list[tuple[np.ndarray, int, int]]:
    """Slice each motion ROI into one or more native-resolution tiles.

    No resizing is ever performed. Each tile is a direct pixel crop at model
    input size — full detail preserved.

    If model_input_type is 'NV12', `frame` is expected to be the raw NV12
    array (shape: (H*3//2, W)) and tiles are sliced as NV12 — skipping CPU
    colorspace conversion entirely. Otherwise standard BGR slicing is used.

    Args:
        frame:                   Hi-res frame. BGR (H, W, 3) or NV12 (H*3//2, W).
        rois:                    ROIs in preview-space (x, y, w, h).
        tile_size:               Model input size (640).
        preview_downscale_factor: Scale to map preview → hi-res coords.
        model_input_type:        'NV12' or 'BGR'.

    Returns:
        List of (tile_array, tile_x, tile_y) in full hi-res frame coordinates.

    """
    is_nv12 = model_input_type == "NV12"

    # CRITICAL FIX: Use the actual buffer capacity
    # If using HbmMsg1080P, frame.shape[1] is 1920.
    # We want to allow tiles to be cut from the full buffer.
    buffer_w = frame.shape[1]
    buffer_h = (frame.shape[0] * 2 // 3) if is_nv12 else frame.shape[0]

    tiles: list[tuple[np.ndarray, int, int]] = []

    def emit(tx: int, ty: int):
        # Ensure the tile start doesn't go negative
        tx = max(0, tx)
        ty = max(0, ty)

        # Ensure we don't slice outside the PHYSICAL shared memory buffer
        if tx + tile_size > buffer_w:
            tx = buffer_w - tile_size
        if ty + tile_size > buffer_h:
            ty = buffer_h - tile_size

        if is_nv12:
            # This is a ZERO-COPY view of the shared memory
            tile = _slice_nv12_tile(frame, buffer_w, buffer_h, tx, ty, tile_size)
        else:
            tile = _slice_bgr_tile(frame, tx, ty, tile_size)
        tiles.append((tile, tx, ty))

    for roi in rois:
        # Map preview ROI to high-res coordinates
        rx = int(roi[0] * preview_downscale_factor)
        ry = int(roi[1] * preview_downscale_factor)
        rw = int(roi[2] * preview_downscale_factor)
        rh = int(roi[3] * preview_downscale_factor)

        # Optimization for RDK X5:
        # If the model wants 640x640 and our motion is inside the 640x352 area,
        # we just emit one tile starting at 0,0.
        # The 'extra' 288 pixels of height will be raw buffer data (Zero-cost padding).
        if rw <= tile_size and rh <= tile_size:
            # Center the tile on the ROI if possible, otherwise 0,0
            emit(rx + rw // 2 - tile_size // 2, ry + rh // 2 - tile_size // 2)
        else:
            # Standard grid logic for larger ROIs
            y = ry
            while True:
                ty = min(y, buffer_h - tile_size)
                x = rx
                while True:
                    tx = min(x, buffer_w - tile_size)
                    emit(tx, ty)
                    if tx >= rx + rw - tile_size or tx >= buffer_w - tile_size:
                        break
                    x += tile_size - int(tile_size * 0.1)
                if ty >= ry + rh - tile_size or ty >= buffer_h - tile_size:
                    break
                y += tile_size - int(tile_size * 0.1)

    return tiles


def get_padded_roi_images(
    *,
    frame: np.ndarray,
    rois: list[Box],
    preview_downscale_factor: float,
    target_size: int,
    pad_color: tuple[int, int, int] = (0, 0, 0),
):
    """Crop ROIs from a frame.

    Preserves their aspect ratio by either center-cropping or padding, and resizes them to a square target size.

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
                start_y : start_y + min_dim,
                start_x : start_x + min_dim,
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
        assert final_image.shape[:2] == (target_size, target_size), "Final image processing failed."

        final_roi_images.append((final_image, scale, effective_origin))

    return final_roi_images
