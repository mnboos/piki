import ctypes
import json
import logging
import os
import time
import traceback
from pathlib import Path

import numpy as np
from hobot_dnn import pyeasy_dnn as dnn

from .shared import worker_ready

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
    model_file = Path(model_file_env).resolve() if model_file_env else Path(__file__).parent / "models" / "yolov10n.bin"

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
    print(f"Model input type detected: {MODEL_INPUT_TYPE} (raw: {_input_type_name})")
    print("done")

    worker_ready.set()

    def detect_objects(image: np.ndarray) -> tuple[int, list]:
        # assert image.shape[0] == IMG_SIZE and image.shape[1] == IMG_SIZE, (
        #     f"Image shape is {image.shape}, but expected ({IMG_SIZE}, {IMG_SIZE})"
        # )

        input_data = np.expand_dims(image, axis=0)

        t0 = time.perf_counter()
        # pyeasy_dnn returns a list of PyDNNTensor, extracting buffer to get numpy arrays
        outputs = model.forward(input_data)

        results = yolov10_post_process(outputs=outputs, score_threshold=0.5)
        tt = round((time.perf_counter() - t0) * 1000)

        logger.debug(f"results: {results}")

        return tt, results
except:
    traceback.print_exc()
    raise
