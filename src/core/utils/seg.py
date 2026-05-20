"""
BPU-accelerated instance segmentation for bbox regions.

Uses the pre-installed YOLO11n-seg model on the RDK X5 BPU via hbm_runtime.
When SEGMENTATION_MODEL_FILE is not set or the model fails to load, falls back
to the motion-mask centroid.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_NAME = ""
_INPUT_SIZE = 640
_INPUT_H = 960  # 640 * 3 // 2
_MODEL_OK = False
_OUTPUT_NAMES: list[str] = []
_OUTPUT_QUANTS: dict = {}
_STRIDES = [8, 16, 32]
_ANCHOR_SIZES = [80, 40, 20]
_REG = 16
_CONF_THRES_RAW = -np.log(1 / 0.25 - 1)  # sigmoid inverse for score=0.25
_LAST_MASK: Optional[np.ndarray] = None   # cache of the last computed mask for display


def _init_seg_model() -> bool:
    """Lazy-load the YOLO11n-seg model via hbm_runtime."""
    global _MODEL, _MODEL_NAME, _MODEL_OK, _OUTPUT_NAMES, _OUTPUT_QUANTS
    if _MODEL_OK:
        return True

    model_path = os.environ.get("SEGMENTATION_MODEL_FILE",
                                 "/app/pydev_demo/03_instance_segmentation_sample/"
                                 "02_ultralytics_yolo11_seg/yolo11n_seg_bayese_640x640_nv12.bin")
    path = Path(model_path)
    if not path.is_file():
        logger.info("Segmentation model not found at %s — segmentation centroid unavailable", path)
        return False

    try:
        import hbm_runtime  # noqa: PLC0415

        # Ensure Horizon postprocess utils are importable.
        _utils_dir = "/app/pydev_demo/utils"
        if _utils_dir not in sys.path:
            sys.path.insert(0, _utils_dir)

        _MODEL = hbm_runtime.HB_HBMRuntime(str(path))
        _MODEL_NAME = _MODEL.model_names[0]
        _OUTPUT_NAMES = list(_MODEL.output_names[_MODEL_NAME])
        _OUTPUT_QUANTS = _MODEL.output_quants[_MODEL_NAME]
        _MODEL_OK = True
        logger.info("Segmentation model loaded: %s (%d outputs)", path, len(_OUTPUT_NAMES))
        return True
    except Exception:
        logger.exception("Failed to load segmentation model")
        return False


def _nv12_crop_to_bgr(nv12: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Extract a BGR image from an NV12 frame using only the Y-plane crop.

    The Y-plane (luma) provides a grayscale image of the region; we convert it
    to a 3-channel BGR for the preprocessing pipeline. This avoids the complexity
    of NV12-to-NV12 resizing and is sufficient for segmentation purposes.
    """
    x1, y1, x2, y2 = bbox
    y_crop = nv12[y1:y2, x1:x2]
    return np.stack([y_crop] * 3, axis=-1)  # grayscale → BGR


def _preprocess(bgr_crop: np.ndarray) -> dict:
    """Convert a BGR image crop to the NV12 tensor format expected by the model."""
    import cv2  # noqa: PLC0415

    from preprocess_utils import resized_image, bgr_to_nv12_planes  # noqa: PLC0415

    resized = resized_image(bgr_crop, _INPUT_SIZE, _INPUT_SIZE, resize_type=1)
    y, uv = bgr_to_nv12_planes(resized)
    nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
    nv12 = nv12.reshape((1, _INPUT_H, _INPUT_SIZE, 1))

    return {_MODEL_NAME: {_MODEL.input_names[_MODEL_NAME][0]: nv12}}


def _postprocess(outputs: dict, crop_w: int, crop_h: int) -> Optional[np.ndarray]:
    """Decode the highest-confidence mask from the model output.

    Returns a binary mask at the crop resolution, or None if no valid detection.
    """
    import cv2  # noqa: PLC0415
    from postprocess_utils import (  # noqa: PLC0415
        dequantize_outputs,
        filter_classification,
        decode_boxes,
        filter_mces,
        NMS,
        decode_masks,
    )

    fp32_outputs = dequantize_outputs(outputs, _OUTPUT_QUANTS)
    protos = fp32_outputs[_OUTPUT_NAMES[9]][0]

    all_dbboxes, all_scores, all_ids, all_mces = [], [], [], []
    weights = np.arange(_REG, dtype=np.float32)[np.newaxis, np.newaxis, :]

    for i, (stride, anchor_size) in enumerate(zip(_STRIDES, _ANCHOR_SIZES)):
        scores, ids, valid = filter_classification(
            fp32_outputs[_OUTPUT_NAMES[3 * i]], _CONF_THRES_RAW)
        if valid.size == 0:
            continue
        all_dbboxes.append(decode_boxes(
            fp32_outputs[_OUTPUT_NAMES[3 * i + 1]], valid, anchor_size, stride, weights))
        all_scores.append(scores)
        all_ids.append(ids)
        all_mces.append(filter_mces(fp32_outputs[_OUTPUT_NAMES[3 * i + 2]], valid))

    if not all_dbboxes:
        return None

    dbboxes = np.concatenate(all_dbboxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    ids = np.concatenate(all_ids, axis=0)
    mces = np.concatenate(all_mces, axis=0)

    keep = NMS(dbboxes, scores, ids, iou_thresh=0.7)
    if not keep:
        return None

    # Take the highest-confidence detection's mask.
    best_idx = keep[np.argmax(scores[keep])]

    masks = decode_masks(
        mces[best_idx:best_idx + 1], dbboxes[best_idx:best_idx + 1], protos,
        _INPUT_SIZE, _INPUT_SIZE, protos.shape[1], protos.shape[0],
        mask_thresh=0.5,
    )

    if not masks:
        return None

    mask = masks[0]
    if mask.sum() < 5:
        return None

    # Resize mask from the detection's bbox-size within model input → crop size.
    mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    # Light morphological open to clean up noise.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    global _LAST_MASK
    _LAST_MASK = mask.copy()

    return mask


def segmentation_centroid(
    bbox_normalized: list[float],
    nv12_frame: np.ndarray,
    frame_shape: tuple[int, int],
) -> Optional[tuple[float, float]]:
    """Compute a segmentation-mask centroid within the detection bbox.

    Args:
        bbox_normalized: [ymin, xmin, ymax, xmax] in [0, 1]
        nv12_frame: Current NV12 frame (H*3//2, W)
        frame_shape: (height, width) of the Y plane

    Returns:
        (cx_n, cy_n) normalised to [0, 1], or None on failure.
    """
    if not _init_seg_model():
        return None

    try:
        fh, fw = frame_shape[:2]
        ymin, xmin, ymax, xmax = bbox_normalized
        x1 = max(0, int(xmin * fw))
        y1 = max(0, int(ymin * fh))
        x2 = min(fw, int(xmax * fw))
        y2 = min(fh, int(ymax * fh))

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 16 or crop_h < 16:
            return None

        bgr = _nv12_crop_to_bgr(nv12_frame, (x1, y1, x2, y2))
        input_tensor = _preprocess(bgr)
        outputs = _MODEL.run(input_tensor)[_MODEL_NAME]
        mask = _postprocess(outputs, crop_w, crop_h)

        if mask is None:
            return None

        ys, xs = np.where(mask > 0)
        if xs.size < 5:
            return None

        cx_local = float(xs.mean())
        cy_local = float(ys.mean())
        return (x1 + cx_local) / fw, (y1 + cy_local) / fh

    except Exception:
        logger.exception("Segmentation centroid failed")
        return None


def get_last_seg_mask() -> Optional[np.ndarray]:
    """Return a copy of the last computed segmentation mask, or None."""
    mask = _LAST_MASK
    return mask.copy() if mask is not None else None
