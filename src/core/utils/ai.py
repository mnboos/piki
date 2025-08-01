import time
import cv2
from ncnn.model_zoo import get_model
import numpy as np
import multiprocessing as mp
from ctypes import c_float

from ncnn.model_zoo.nanodet import NanoDet
from ncnn.utils import Detect_Object

prob_threshold = mp.Value(c_float, 0.5)

models = {"nanodet", "mobilenetv2_ssdlite", "mobilenetv3_ssdlite", ""}

print("Loading model...")
net: NanoDet = get_model(
    "nanodet",
    # target_size=480,
    # prob_threshold=prob_threshold.value,
    # nms_threshold=0.1,
    # num_threads=1,
    use_gpu=False,
)

print("Model loaded.")


def detect_objects(image_data: np.ndarray):
    # return []
    # image_copy = image_data[:]
    # image_copy = image_data

    # print("start object detection...")

    # image_np = np.frombuffer(image_data, np.uint8)
    # image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    # image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # 3. Perform object detection and measure inference time
    print("Starting inference...")
    start_time = time.perf_counter()

    objects: list[Detect_Object] = net(image_data)  # This is the line we are timing
    # objects = []

    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000
    worker_pid = mp.current_process().pid
    print(
        f"[Worker-{worker_pid}] Inference complete in {inference_time_ms:.2f} ms.",
    )

    #
    # threshold = prob_threshold.value
    #
    # # 4. Print the detection results as text
    # if objects:
    # print("\n--- Object Detection Results ---")
    results = []
    for obj in objects:
        class_name = net.class_names[int(obj.label)]
        # class_name = (
        #     COCO_CLASSES[obj.label]
        #     if obj.label < len(COCO_CLASSES)
        #     else f"Unknown_ID_{obj.label}"
        # )
        # bbox = obj.rect

        results.append((obj.prob, class_name, obj.rect))

        # print(f"""Object {i + 1}:"
        #   - Label: {class_name}
        #   - Confidence: {obj.prob:.2%}
        #   - Bounding Box (x, y, width, height): ({bbox.x}, {bbox.y}, {bbox.w}, {bbox.h})\n""")
    return results
