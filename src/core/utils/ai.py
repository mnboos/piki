import time
import cv2
from ncnn.model_zoo import get_model
import numpy as np


print("Loading model...")
net = get_model(
    "nanodet",
    target_size=320,
    prob_threshold=0.2,
    nms_threshold=0.5,
    num_threads=1,
    use_gpu=False,
)
print("Model loaded.")


COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
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
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def inference(image_data):
    assert image_data

    image_copy = image_data[:]

    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # 3. Perform object detection and measure inference time
    print("\nStarting inference...")
    start_time = time.perf_counter()

    objects = net(image)  # This is the line we are timing

    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000
    print(f"Inference complete in {inference_time_ms:.2f} ms.")

    results = []

    # 4. Print the detection results as text
    if objects:
        print("\n--- Object Detection Results ---")
        for i, obj in enumerate(objects):
            class_name = (
                COCO_CLASSES[obj.label]
                if obj.label < len(COCO_CLASSES)
                else f"Unknown_ID_{obj.label}"
            )
            bbox = obj.rect

            results.append((obj.prob, class_name, bbox))

            print(f"Object {i + 1}:")
            print(f"  - Label: {class_name}")
            print(f"  - Confidence: {obj.prob:.2%}")
            print(
                f"  - Bounding Box (x, y, width, height): ({bbox.x}, {bbox.y}, {bbox.w}, {bbox.h})"
            )
        print("------------------------------")
    else:
        print("\nNo objects detected in the image.")
    return image_copy, results
