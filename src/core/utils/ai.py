import ctypes
import json
import logging
import os
import time
import traceback
from pathlib import Path

import numpy as np
from hobot_dnn import pyeasy_dnn as dnn

from .shared import prob_threshold, worker_ready

logger = logging.getLogger(__name__)

QUANTIZE_ON = True

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

# CLASSES = (
#     "person",
#     "bicycle",
#     "car",
#     "motorbike ",
#     "aeroplane ",
#     "bus ",
#     "train",
#     "truck ",
#     "boat",
#     "traffic light",
#     "fire hydrant",
#     "stop sign ",
#     "parking meter",
#     "bench",
#     "bird",
#     "cat",
#     "dog ",
#     "horse ",
#     "sheep",
#     "cow",
#     "elephant",
#     "bear",
#     "zebra ",
#     "giraffe",
#     "backpack",
#     "umbrella",
#     "handbag",
#     "tie",
#     "suitcase",
#     "frisbee",
#     "skis",
#     "snowboard",
#     "sports ball",
#     "kite",
#     "baseball bat",
#     "baseball glove",
#     "skateboard",
#     "surfboard",
#     "tennis racket",
#     "bottle",
#     "wine glass",
#     "cup",
#     "fork",
#     "knife ",
#     "spoon",
#     "bowl",
#     "banana",
#     "apple",
#     "sandwich",
#     "orange",
#     "broccoli",
#     "carrot",
#     "hot dog",
#     "pizza ",
#     "donut",
#     "cake",
#     "chair",
#     "sofa",
#     "pottedplant",
#     "bed",
#     "diningtable",
#     "toilet ",
#     "tvmonitor",
#     "laptop	",
#     "mouse	",
#     "remote ",
#     "keyboard ",
#     "cell phone",
#     "microwave ",
#     "oven ",
#     "toaster",
#     "sink",
#     "refrigerator ",
#     "book",
#     "clock",
#     "vase",
#     "scissors ",
#     "teddy bear ",
#     "hair drier",
#     "toothbrush ",
# )

# You still need your list of class names from the COCO dataset
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


libpostprocess = ctypes.CDLL("/usr/lib/libpostprocess.so")


class hbSysMem_t(ctypes.Structure):
    _fields_ = [("phyAddr", ctypes.c_double), ("virAddr", ctypes.c_void_p), ("memSize", ctypes.c_int)]


class hbDNNQuantiShift_yt(ctypes.Structure):
    _fields_ = [("shiftLen", ctypes.c_int), ("shiftData", ctypes.c_char_p)]


class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen", ctypes.c_int),
        ("scaleData", ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen", ctypes.c_int),
        ("zeroPointData", ctypes.c_char_p),
    ]


class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [("dimensionSize", ctypes.c_int * 8), ("numDimensions", ctypes.c_int)]


class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape", hbDNNTensorShape_t),
        ("alignedShape", hbDNNTensorShape_t),
        ("tensorLayout", ctypes.c_int),
        ("tensorType", ctypes.c_int),
        ("shift", hbDNNQuantiShift_yt),
        ("scale", hbDNNQuantiScale_t),
        ("quantiType", ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize", ctypes.c_int),
        ("stride", ctypes.c_int * 8),
    ]


class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [("sysMem", hbSysMem_t * 4), ("properties", hbDNNTensorProperties_t)]


class Yolov5PostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height", ctypes.c_int),
        ("width", ctypes.c_int),
        ("ori_height", ctypes.c_int),
        ("ori_width", ctypes.c_int),
        ("score_threshold", ctypes.c_float),
        ("nms_threshold", ctypes.c_float),
        ("nms_top_k", ctypes.c_int),
        ("is_pad_resize", ctypes.c_int),
    ]


get_Postprocess_result = libpostprocess.Yolov5PostProcess
get_Postprocess_result.argtypes = [ctypes.POINTER(Yolov5PostProcessInfo_t)]
get_Postprocess_result.restype = ctypes.c_char_p


def get_TensorLayout(layout: str):
    return 2 if layout == "NCHW" else 0


# ---------------------------------------------------------------------------
# YOLOv8 / YOLOv12n post-processor (pure numpy, ~5ms)
# Output format: 6 tensors alternating cls(float32) / box-DFL(int32) at 3 scales
# ---------------------------------------------------------------------------
_YV8_REG_MAX = 16
_YV8_BINS = np.arange(_YV8_REG_MAX, dtype=np.float32)


def yolov8_post_process(*, outputs, img_w=640, img_h=640, conf_thres=0.5, iou_thres=0.45):
    logit_thres = np.log(conf_thres / (1.0 - conf_thres))
    all_boxes, all_confs, all_cls = [], [], []

    for si, stride in enumerate([8, 16, 32]):
        cls_raw = outputs[si * 2].buffer.squeeze(0)       # (H, W, 80) float32 logits
        box_int = outputs[si * 2 + 1].buffer.squeeze(0)  # (H, W, 64) int32 quantized DFL
        scale = outputs[si * 2 + 1].properties.scale_data  # 4 floats, one per ltrb direction

        # Pre-filter: only process grid cells with a confident detection
        mask = cls_raw.max(axis=-1) > logit_thres
        if not mask.any():
            continue

        cls_scores = 1.0 / (1.0 + np.exp(-cls_raw[mask]))  # sigmoid, shape (N, 80)

        # Dequantize + DFL softmax weighted sum → ltrb in pixels
        box_f = box_int[mask].astype(np.float32).reshape(-1, 4, _YV8_REG_MAX)  # (N, 4, 16)
        for d in range(4):
            box_f[:, d, :] *= scale[d]
        box_f -= box_f.max(-1, keepdims=True)
        np.exp(box_f, out=box_f)
        box_f /= box_f.sum(-1, keepdims=True)
        ltrb = (box_f * _YV8_BINS).sum(-1) * stride  # (N, 4)

        ys, xs = np.where(mask)
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        x1 = np.clip(cx - ltrb[:, 0], 0, img_w)
        y1 = np.clip(cy - ltrb[:, 1], 0, img_h)
        x2 = np.clip(cx + ltrb[:, 2], 0, img_w)
        y2 = np.clip(cy + ltrb[:, 3], 0, img_h)

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_confs.append(cls_scores.max(-1))
        all_cls.append(cls_scores.argmax(-1).astype(np.int32))

    if not all_boxes:
        return []
    boxes = np.concatenate(all_boxes)
    confs = np.concatenate(all_confs)
    cls_ids = np.concatenate(all_cls)

    results = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        b, s = boxes[idx], confs[idx]
        order = s.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ix1 = np.maximum(b[i, 0], b[order[1:], 0])
            iy1 = np.maximum(b[i, 1], b[order[1:], 1])
            ix2 = np.minimum(b[i, 2], b[order[1:], 2])
            iy2 = np.minimum(b[i, 3], b[order[1:], 3])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            union = ((b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
                     + (b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1])
                     - inter)
            order = order[1:][inter / np.maximum(union, 1e-6) < iou_thres]
        for k in keep:
            label = CLASSES[int(c)].strip() if int(c) < len(CLASSES) else str(int(c))
            results.append((label, float(s[k]), b[k]))
    return results


def yolov10_post_process(*, outputs, img_size=640, score_threshold=0.25):
    yolov5_postprocess_info = Yolov5PostProcessInfo_t()
    yolov5_postprocess_info.height = img_size
    yolov5_postprocess_info.width = img_size
    yolov5_postprocess_info.ori_height = img_size
    yolov5_postprocess_info.ori_width = img_size
    yolov5_postprocess_info.score_threshold = 0.4
    yolov5_postprocess_info.nms_threshold = 0.45
    yolov5_postprocess_info.nms_top_k = 20
    yolov5_postprocess_info.is_pad_resize = 0

    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
    for i in range(len(models[0].outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(outputs[i].properties.layout)
        # print(output_tensors[i].properties.tensorLayout)
        if len(outputs[i].properties.scale_data) == 0:
            output_tensors[i].properties.quantiType = 0
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(
                outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p
            )
        else:
            output_tensors[i].properties.quantiType = 2
            output_tensors[i].properties.scale.scaleData = outputs[i].properties.scale_data.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            )
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(
                outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p
            )

        for j in range(len(outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.dimensionSize[j] = outputs[i].properties.shape[j]

        libpostprocess.Yolov5doProcess(output_tensors[i], ctypes.pointer(yolov5_postprocess_info), i)

    result_str = get_Postprocess_result(ctypes.pointer(yolov5_postprocess_info)).decode("utf-8")

    data = json.loads(result_str[16:])  # strip "YOLOV5_RESULT:" prefix

    results = []
    for det in data:
        label = CLASSES[det["id"]] if det["id"] < len(CLASSES) else str(det["id"])
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        results.append((label.strip(), float(det["score"]), np.array(bbox)))
    return results


try:
    print("Loading model...")

    model_file_env = os.environ.get("MODEL_FILE")
    model_file = (
        Path(model_file_env).resolve()
        if model_file_env
        else Path("/app/model/basic/yolov8_640x640_nv12.bin")
    )

    assert model_file.is_file(), f"Model file {model_file} not found!"

    # Load Hobot model
    print("--> Loading model via pyeasy_dnn")
    models = dnn.load(str(model_file.absolute()))
    model = models[0]

    # Detect whether the model was compiled with NV12 input or BGR/RGB.
    # NV12 models skip CPU colorspace conversion — the BPU handles it internally.
    _input_type = model.inputs[0].properties.tensor_type
    _input_type_name = str(_input_type)
    MODEL_INPUT_TYPE = "NV12" if "NV12" in _input_type_name.upper() or "YUV" in _input_type_name.upper() else "BGR"
    # YOLOv8 / YOLOv12n have 6 output tensors (alternating cls/box at 3 scales)
    _USE_YOLOv8_DECODER = len(model.outputs) == 6
    print(f"Model input type detected: {MODEL_INPUT_TYPE} (raw: {_input_type_name})")
    print(f"Post-processor: {'yolov8 (numpy DFL)' if _USE_YOLOv8_DECODER else 'yolov5 (libpostprocess)'}")
    print("done")

    worker_ready.set()

    def detect_objects(image: np.ndarray) -> tuple[int, list]:
        _profile = bool(os.environ.get("PIKI_PROFILE"))

        t_fwd = time.perf_counter()
        outputs = model.forward(image)
        if _profile:
            logger.info("PERF stage=bpu_forward ms=%.2f", (time.perf_counter() - t_fwd) * 1000)

        conf = prob_threshold.value
        t_dec = time.perf_counter()
        if _USE_YOLOv8_DECODER:
            results = yolov8_post_process(outputs=outputs, conf_thres=conf)
        else:
            results = yolov10_post_process(outputs=outputs, score_threshold=conf)
        if _profile:
            logger.info("PERF stage=yolov8_decode ms=%.2f", (time.perf_counter() - t_dec) * 1000)

        tt = round((time.perf_counter() - t_fwd) * 1000)
        logger.debug(f"results: {results}")
        return tt, results
except:
    traceback.print_exc()
    raise
