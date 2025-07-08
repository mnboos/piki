import cv2
import ncnn
from ncnn.model_zoo import get_model
import os
import time  # Import the time module

# --- Create a placeholder for cat.jpg if it doesn't exist ---
if not os.path.exists("cat.jpg"):
    print("Creating a placeholder 'cat.jpg'. Please replace with your own image.")
    import numpy as np
    dummy_cat_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_cat_image, "Placeholder for cat.jpg", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite("cat.jpg", dummy_cat_image)

# --- COCO Class Names ---
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# --- Main Script ---
try:
    # 1. Load the NCNN model (NanoDet)
    print("Loading model...")
    net = get_model(
        "nanodet",
        target_size=240,
        prob_threshold=0.5,
        nms_threshold=0.5,
        num_threads=4,
        use_gpu=False,
    )
    print("Model loaded.")

    # 2. Load the image
    image_path = "cat.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Could not read the image file: {image_path}. Please ensure it exists and is a valid image.")

    # 3. Perform object detection and measure inference time
    print("\nStarting inference...")
    start_time = time.perf_counter()
    
    objects = net(image) # This is the line we are timing
    
    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000
    print(f"Inference complete in {inference_time_ms:.2f} ms.")


    # 4. Print the detection results as text
    if objects:
        print("\n--- Object Detection Results ---")
        for i, obj in enumerate(objects):
            class_name = COCO_CLASSES[obj.label] if obj.label < len(COCO_CLASSES) else f"Unknown_ID_{obj.label}"
            bbox = obj.rect

            print(f"Object {i+1}:")
            print(f"  - Label: {class_name}")
            print(f"  - Confidence: {obj.prob:.2%}")
            print(f"  - Bounding Box (x, y, width, height): ({bbox.x}, {bbox.y}, {bbox.w}, {bbox.h})")
        print("------------------------------")
    else:
        print("\nNo objects detected in the image.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure you have run 'pip install ncnn-python opencv-python numpy'")
