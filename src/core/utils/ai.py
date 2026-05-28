import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from hbm_runtime import HB_HBMRuntime

from .shared import prob_threshold, prob_threshold_keep

logger = logging.getLogger(__name__)

IMG_SIZE = 640
NMS_THRESH = 0.45

CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorbike ",
    "aeroplane ",
    "bus ",
    "train",
    "truck ",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign ",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog ",
    "horse ",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra ",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife ",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza ",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet ",
    "tvmonitor",
    "laptop	",
    "mouse	",
    "remote ",
    "keyboard ",
    "cell phone",
    "microwave ",
    "oven ",
    "toaster",
    "sink",
    "refrigerator ",
    "book",
    "clock",
    "vase",
    "scissors ",
    "teddy bear ",
    "hair drier",
    "toothbrush ",
)


# ---------------------------------------------------------------------------
# Inline YOLO26 post-processing — adapted from RDK model zoo post_utils.
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _filter_classification(cls_output: np.ndarray, conf_thres_raw: float):
    """Threshold raw logits, apply sigmoid, return (scores, class_ids, flat_indices)."""
    h, w, c = cls_output.shape
    flat = cls_output.reshape(-1, c)
    max_raw = flat.max(axis=-1)
    valid = np.where(max_raw >= conf_thres_raw)[0]
    if not valid.size:
        return np.empty(0), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    ids = flat[valid].argmax(axis=-1).astype(np.int32)
    scores = _sigmoid(max_raw[valid])
    return scores, ids, valid


def _decode_ltrb_boxes(flat_indices: np.ndarray, ltrb: np.ndarray,
                        stride: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Convert LTRB deltas + grid-centre anchors → xyxy pixel boxes."""
    ys = (flat_indices // grid_w + 0.5) * stride
    xs = (flat_indices % grid_w + 0.5) * stride
    valid_ltrb = ltrb.reshape(-1, 4)[flat_indices] * stride
    x1 = np.clip(xs - valid_ltrb[:, 0], 0, IMG_SIZE)
    y1 = np.clip(ys - valid_ltrb[:, 1], 0, IMG_SIZE)
    x2 = np.clip(xs + valid_ltrb[:, 2], 0, IMG_SIZE)
    y2 = np.clip(ys + valid_ltrb[:, 3], 0, IMG_SIZE)
    return np.stack([x1, y1, x2, y2], axis=1)


def _nms_per_class(boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray,
                    iou_thres: float) -> list[int]:
    """Per-class greedy NMS — returns flat list of kept indices."""
    kept: list[int] = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        order = idx[np.argsort(-scores[idx])]
        while len(order):
            i = int(order[0])
            kept.append(i)
            if len(order) == 1:
                break
            b = boxes[i]
            rest = boxes[order[1:]]
            ix1 = np.maximum(b[0], rest[:, 0])
            iy1 = np.maximum(b[1], rest[:, 1])
            ix2 = np.minimum(b[2], rest[:, 2])
            iy2 = np.minimum(b[3], rest[:, 3])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            area_i = (b[2] - b[0]) * (b[3] - b[1])
            area_o = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
            iou = inter / (area_i + area_o - inter + 1e-9)
            order = order[1:][iou < iou_thres]
    return kept


# ---------------------------------------------------------------------------
# Segmentation mask helpers
# ---------------------------------------------------------------------------

def _crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Zero out mask pixels outside the corresponding bounding box.

    masks: (N, H, W) float32
    boxes: (N, 4) float32 [x1, y1, x2, y2] in mask-space coords
    """
    n, h, w = masks.shape
    x1 = boxes[:, 0].reshape(n, 1, 1)
    y1 = boxes[:, 1].reshape(n, 1, 1)
    x2 = boxes[:, 2].reshape(n, 1, 1)
    y2 = boxes[:, 3].reshape(n, 1, 1)
    r = np.arange(w, dtype=np.float32).reshape(1, 1, w)
    c = np.arange(h, dtype=np.float32).reshape(1, h, 1)
    return masks * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))


def _process_mask(
    protos: np.ndarray,
    masks_in: np.ndarray,
    bboxes: np.ndarray,
    shape: tuple[int, int],
    upsample: bool = True,
) -> np.ndarray:
    """Decode mask coefficients → (N, H, W) boolean masks.

    protos:   (32, mh, mw)
    masks_in: (N, 32)
    bboxes:   (N, 4) [x1,y1,x2,y2] in image-space pixels
    shape:    (img_h, img_w)
    """
    c, mh, mw = protos.shape
    ih, iw = shape
    # (N, 32) × (32, mh*mw) → sigmoid → (N, mh, mw)
    masks = _sigmoid(masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    # Scale bboxes down to proto resolution and crop
    scale = np.array([mw / iw, mh / ih, mw / iw, mh / ih], dtype=np.float32)
    masks = _crop_mask(masks, bboxes * scale)
    if upsample:
        masks = np.stack([
            cv2.resize(m.astype(np.float32), (iw, ih), interpolation=cv2.INTER_LINEAR)
            for m in masks
        ])
    return masks > 0.5


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------
# The BPU model is loaded lazily on the first detect_objects() call so that
# stream startup (import → rclpy.init → camera spin) is not blocked by the
# 5-15s HB_HBMRuntime constructor on cold boot.

_model_file_env = os.environ.get("MODEL_FILE")
if _model_file_env:
    _model_file = Path(_model_file_env).resolve()
else:
    _variant = os.environ.get("YOLO_VARIANT", "n")
    _model_file = (
        Path(__file__).parents[3]
        / "model"
        / f"yolo26{_variant}_seg_bayese_640x640_nv12.bin"
    )

# MODEL_INPUT_TYPE is derived from the filename alone so it's available at
# import time without touching the BPU. The filename is authoritative.
MODEL_INPUT_TYPE = "NV12" if "nv12" in str(_model_file).lower() else "BGR"

_runtime: "HB_HBMRuntime | None" = None
_model_name: str = ""
_input_name: str = ""
_output_names: list[str] = []
_sorted_output_names: list[str] = []   # populated on first inference
_model_lock = threading.Lock()

# Within each stride group the last-dim order is: cls(80) → box(4) → mc(32).
_C_ORDER: dict[int, int] = {80: 0, 4: 1, 32: 2}


def _sort_outputs(raw_outputs: dict) -> list[str]:
    """Return output names in the canonical seg order:
    [cls0, box0, mc0, cls1, box1, mc1, cls2, box2, mc2, proto]
    Detection heads: shape (1,H,W,C) with H∈{20,40,80}, C∈{4,32,80}.
    Proto:           shape (1,H,W,32) with H=160 (largest spatial dimension).
    Sorted H descending so si=0 → stride-8 (H=80), si=1 → stride-16, si=2 → stride-32.
    """
    info: list[tuple[int, int, str]] = []
    for name, arr in raw_outputs.items():
        a = arr.squeeze(0)
        h = int(a.shape[0])
        c = int(a.shape[-1])
        info.append((h, c, name))
    max_h = max(x[0] for x in info)
    det = [(h, c, name) for h, c, name in info if h < max_h]
    proto = [name for h, c, name in info if h == max_h]
    # Largest H first (stride-8 first), within a stride group: cls, box, mc
    det.sort(key=lambda x: (-x[0], _C_ORDER.get(x[1], 9)))
    return [name for _, _, name in det] + proto


def _init_model() -> None:
    """Load the YOLO model onto the BPU — expensive, called once on first inference."""
    global _runtime, _model_name, _input_name, _output_names
    if _runtime is not None:
        return
    with _model_lock:
        if _runtime is not None:
            return
        assert _model_file.is_file(), f"Model file {_model_file} not found!"
        print(f"Loading model {_model_file.name} via hbm_runtime...")
        _runtime = HB_HBMRuntime(str(_model_file.absolute()))
        _model_name = next(iter(_runtime.input_names))
        _input_name = _runtime.input_names[_model_name][0]
        _output_names = _runtime.output_names[_model_name]
        print(f"Model loaded. input type: {MODEL_INPUT_TYPE}")


def detect_objects(image: np.ndarray) -> tuple[int, list]:
    """Run inference on a single 640×640 tile; return ``(elapsed_ms, detections)``.

    Each detection is a 5-tuple:
    ``(label_str, confidence, xyxy_px_array, centroid_tile_px, polygon_tile_px_list)``

    * ``xyxy_px_array``        – ``np.array([x1,y1,x2,y2])`` in 640×640 tile pixel coords
    * ``centroid_tile_px``     – ``(cx, cy)`` float tile-pixel coords, or ``None`` if the
                                  instance mask contains no foreground pixels
    * ``polygon_tile_px_list`` – list of contours; each contour is a flat ``[x1,y1,...]``
                                  list in tile pixel coords, or ``None``
    """
    global _sorted_output_names
    _init_model()  # no-op after first call

    t0 = time.perf_counter()
    outputs = _runtime.run({_model_name: {_input_name: image}})
    outputs = outputs[_model_name]

    # Sort outputs into canonical order on the very first inference.
    if not _sorted_output_names:
        _sorted_output_names = _sort_outputs(outputs)
        logger.debug("Seg output order resolved: %s", _sorted_output_names)

    # Use the lower "keep" threshold so on_done()'s hysteresis still
    # receives low-confidence candidates.
    p_val = prob_threshold.value
    pk_val = prob_threshold_keep.value
    conf = min(p_val, pk_val)
    conf_raw = -np.log(1.0 / max(conf, 1e-6) - 1.0)

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_cls: list[np.ndarray] = []
    all_mc: list[np.ndarray] = []
    max_logits_per_stride: list[float] = []
    n_above_per_stride: list[int] = []

    for si, stride in enumerate([8, 16, 32]):
        cls_out = outputs[_sorted_output_names[si * 3]].squeeze(0)       # (H, W, 80)
        box_out = outputs[_sorted_output_names[si * 3 + 1]].squeeze(0)   # (H, W, 4)
        mc_out  = outputs[_sorted_output_names[si * 3 + 2]].squeeze(0)   # (H, W, 32)
        gh, gw = cls_out.shape[:2]

        cls_flat = cls_out.reshape(-1, cls_out.shape[-1])
        max_raw = cls_flat.max(axis=-1)
        max_logits_per_stride.append(float(max_raw.max()) if max_raw.size else -999)
        n_above_per_stride.append(int((max_raw >= conf_raw).sum()))

        scores, ids, valid = _filter_classification(cls_out, conf_raw)
        if not valid.size:
            continue
        boxes = _decode_ltrb_boxes(valid, box_out, stride, gh, gw)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_cls.append(ids)
        all_mc.append(mc_out.reshape(-1, 32)[valid])

    if not all_boxes:
        img_stats = f"min={image.min()} max={image.max()} mean={image.mean():.1f}" if image.size else "empty"
        logger.warning(
            "AI_NO_DETS p_val=%.4f pk_val=%.4f conf=%.4f conf_raw=%.2f "
            "max_logits=%s n_above=%s img=(%s)",
            p_val, pk_val, conf, conf_raw,
            ["%.2f" % v for v in max_logits_per_stride],
            n_above_per_stride,
            img_stats,
        )
        return round((time.perf_counter() - t0) * 1000), []

    boxes    = np.concatenate(all_boxes)
    scores   = np.concatenate(all_scores)
    cls_ids  = np.concatenate(all_cls)
    mc       = np.concatenate(all_mc)          # (N_pre, 32)

    indices     = _nms_per_class(boxes, scores, cls_ids, NMS_THRESH)
    kept_boxes  = boxes[indices].astype(np.float32)
    kept_scores = scores[indices]
    kept_cls    = cls_ids[indices]
    kept_mc     = mc[indices].astype(np.float32)   # (N, 32)

    # Decode proto and generate per-instance binary masks.
    proto_raw = outputs[_sorted_output_names[9]].squeeze(0)               # (160, 160, 32)
    proto = np.ascontiguousarray(proto_raw.transpose(2, 0, 1).astype(np.float32))  # (32, 160, 160)
    masks = _process_mask(proto, kept_mc, kept_boxes, (IMG_SIZE, IMG_SIZE))  # (N, 640, 640) bool

    results: list = []
    for i in range(len(indices)):
        label    = CLASSES[kept_cls[i]].strip()
        conf_val = float(kept_scores[i])
        box      = kept_boxes[i]
        mask     = masks[i]  # (640, 640) bool

        ys, xs = np.where(mask)
        if xs.size > 0:
            centroid_px: "tuple[float,float] | None" = (float(xs.mean()), float(ys.mean()))

            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon: "list[list[float]] | None" = []
            for cnt in contours:
                eps = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                if len(approx) >= 3:
                    polygon.append(approx.reshape(-1).tolist())
            if not polygon:
                polygon = None
        else:
            centroid_px = None
            polygon = None

        results.append((label, conf_val, box, centroid_px, polygon))

    return round((time.perf_counter() - t0) * 1000), results
