import os
import time
import traceback
from pathlib import Path

import numpy as np
from hobot_dnn import pyeasy_dnn as dnn

from .shared import worker_ready

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


def yolov10_post_process(outputs: list, confidence_threshold: float = 0.5):
    """
    Robustly parses YOLOv10 output tensors from the RDK X5 BPU.
    """
    # 1. Access the first output buffer
    detections = outputs[0]

    # 2. SQUEEZE: This is the critical fix.
    # If shape is (1, 1, 300, 6) or (1, 300, 6), this turns it into (300, 6)
    detections = np.squeeze(detections)

    # 3. Handle the case where the model finds exactly 0 or 1 object
    if detections.ndim == 1:
        # If shape is (6,), it's a single detection; make it (1, 6)
        detections = np.expand_dims(detections, axis=0)
    elif detections.ndim == 0:
        return []

    final_results = []
    for detection in detections:
        # detection is now guaranteed to be [x1, y1, x2, y2, score, label_idx]
        score = float(detection[4])

        if score >= confidence_threshold:
            label_idx = int(detection[5])
            if label_idx < len(CLASSES):
                label = CLASSES[label_idx]
                box = detection[0:4] # [x1, y1, x2, y2]
                final_results.append((label, score, box))

    return final_results
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
        hobot_outputs = model.forward(input_data)
        outputs = [out.buffer for out in hobot_outputs]

        results = yolov10_post_process(outputs, confidence_threshold=0.5)
        tt = round((time.perf_counter() - t0) * 1000)

        return tt, results
except:
    traceback.print_exc()
    raise
