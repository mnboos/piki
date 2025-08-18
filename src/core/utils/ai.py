import time
from pathlib import Path

from ai_edge_litert.interpreter import Interpreter
import ai_edge_litert.interpreter as interpreter
import traceback
import numpy as np

from .shared import prob_threshold, worker_ready

# model_file = "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite"
# model_file = "ssd-mobilenet-v2-tflite-100-int8-default-v1.tflite"
# model_file = "ssd-mobilenet-v2-tflite-fpn-100-uint8-default-v1.tflite"
model_file = "efficientdet-tflite-lite0-int8-v1.tflite"

COCO_LABELS = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}
try:
    print("Loading model...")
    model_path = str(Path(__file__).parent / model_file)
    is_quant = "quant" in model_file or "int8" in model_file

    # armnn_delegate = interpreter.load_delegate(
    #     library="/home/martin/Downloads/ArmNN-linux-x86_64/delegate/libarmnnDelegate.so",
    #     options={
    #         "backends": "GpuAcc,CpuAcc",  # Prioritize GPU, fallback to CPU
    #         "logging-severity": "info",  # Optional: for verbose logging
    #     },
    # )
    # print("Arm NN delegate loaded successfully.")
    # delegates_list = [armnn_delegate]
    #
    # interpreter = Interpreter(
    #     model_path=model_path, experimental_delegates=delegates_list
    # )

    interpreter = Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    inp_id = input_details[0]["index"]
    output_details = interpreter.get_output_details()

    height = 320
    width = 320

    interpreter.resize_tensor_input(input_details[0]["index"], (1, height, width, 3))
    interpreter.allocate_tensors()  # Must re-allocate tensors after resizing

    out_id0 = output_details[0]["index"]
    out_id1 = output_details[1]["index"]
    out_id2 = output_details[2]["index"]
    out_id3 = output_details[3]["index"]

    worker_ready.set()

    def detect_objects(image: np.ndarray) -> tuple[int, list]:
        assert image.shape[0] == width and image.shape[1] == height, (
            f"Image shape is {image.shape}, but expected ({width}, {height})"
        )

        input_data = np.expand_dims(image, axis=0)

        t0 = time.perf_counter()
        interpreter.set_tensor(inp_id, input_data)
        interpreter.invoke()
        tt = round((time.perf_counter() - t0) * 1000)

        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]
        num_det = interpreter.get_tensor(output_details[3]["index"])[0]

        num_objects_found = int(num_det)

        results = []
        threshold = prob_threshold.value
        for i in range(num_objects_found):
            confidence = round(float(scores[i]), 3)
            coco_class_id = int(classes[i])
            if (
                coco_class_id
                and coco_class_id in COCO_LABELS
                and confidence >= threshold
            ):
                # Get bounding box coordinates and scale them to the original image size
                ymin, xmin, ymax, xmax = boxes[i]
                ymin = max(0.0, ymin)
                xmin = max(0.0, xmin)
                ymax = min(1.0, ymax)
                xmax = min(1.0, xmax)

                x = int(xmin * width)
                y = int(ymin * height)
                w = int((xmax - xmin) * width)
                h = int((ymax - ymin) * height)

                # Get the object name from the label map
                label = COCO_LABELS[coco_class_id]
                # print("label: ", label, confidence)
                results.append((label, confidence, (x, y, w, h)))

        return tt, results
except:
    traceback.print_exc()
    raise
