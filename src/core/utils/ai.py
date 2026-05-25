import logging
import os
import time
import traceback
from pathlib import Path

import numpy as np
from hbm_runtime import HB_HBMRuntime

from .shared import prob_threshold, prob_threshold_keep, worker_ready

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
# Model loading + inference
# ---------------------------------------------------------------------------

try:
    print("Loading model...")

    model_file_env = os.environ.get("MODEL_FILE")
    model_file = (
        Path(model_file_env).resolve()
        if model_file_env
        else Path("/app/model/basic/yolo26n_detect_bayese_640x640_nv12.bin")
    )

    assert model_file.is_file(), f"Model file {model_file} not found!"

    print("--> Loading model via hbm_runtime")
    _runtime = HB_HBMRuntime(str(model_file.absolute()))
    _model_name = next(iter(_runtime.input_names))
    _input_name = _runtime.input_names[_model_name][0]
    _output_names = _runtime.output_names[_model_name]

    # Detect NV12 vs BGR input from model metadata.
    _input_type = str(_runtime.input_type[_model_name]) if hasattr(_runtime, "input_type") else ""
    # The HBM runtime metadata may omit or misreport the format; the model
    # binary filename is authoritative (yolo26n_detect_bayese_640x640_nv12.bin).
    MODEL_INPUT_TYPE = "NV12" if "NV12" in _input_type.upper() or "YUV" in _input_type.upper() or "nv12" in str(model_file).lower() else "BGR"
    print(f"Model input type: {MODEL_INPUT_TYPE} (raw: {_input_type}, file: {model_file.name})")
    print("done")

    worker_ready.set()

    def detect_objects(image: np.ndarray) -> tuple[int, list]:
        """Run inference on a single 640×640 tile; return (elapsed_ms, detections).

        Each detection is ``(label_str, confidence, np.array([x1,y1,x2,y2]))``
        in pixel coordinates relative to the 640×640 tile.
        """
        t0 = time.perf_counter()
        outputs = _runtime.run({_model_name: {_input_name: image}})
        outputs = outputs[_model_name]

        # Use the lower "keep" threshold so on_done()'s hysteresis still
        # receives low-confidence candidates.
        p_val = prob_threshold.value
        pk_val = prob_threshold_keep.value
        conf = min(p_val, pk_val)
        conf_raw = -np.log(1.0 / max(conf, 1e-6) - 1.0)

        all_boxes, all_scores, all_cls = [], [], []
        max_logits_per_stride: list[float] = []
        n_above_per_stride: list[int] = []
        for si, stride in enumerate([8, 16, 32]):
            cls_out = outputs[_output_names[si * 2]].squeeze(0)      # (H, W, 80)
            box_out = outputs[_output_names[si * 2 + 1]].squeeze(0)  # (H, W, 4)
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

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        cls_ids = np.concatenate(all_cls)
        indices = _nms_per_class(boxes, scores, cls_ids, NMS_THRESH)
        results = [
            (CLASSES[cls_ids[i]].strip(), float(scores[i]), boxes[i])
            for i in indices
        ]
        tt = round((time.perf_counter() - t0) * 1000)
        return tt, results

except Exception:
    traceback.print_exc()
    raise
