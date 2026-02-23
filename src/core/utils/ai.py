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
    """Parse the output of a YOLOv10 model.

    Args:
        outputs: The list of NumPy arrays from rknn.inference().
                 Assumes a single output tensor of shape (1, 300, 6).
        confidence_threshold: The minimum score for a detection to be kept.

    Returns:
        A list of tuples, where each tuple is (label, confidence, box).
        The box is in [x1, y1, x2, y2] format.

    """
    # The output from rknn.inference() is a list of arrays. YOLOv10 typically has one output.
    detections = outputs[0]

    # The shape is (1, 300, 6). We remove the first dimension (batch size).
    detections = detections[0]  # Shape is now (300, 6)

    final_results = []
    for detection in detections:
        # detection is a row: [x1, y1, x2, y2, score, label_index]
        score = detection[4]

        # Apply the confidence threshold
        if score >= confidence_threshold:
            label_index = int(detection[5])
            box = detection[0:4]  # The box is already in [x1, y1, x2, y2] format

            # Get the class name
            label = CLASSES[label_index]

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
