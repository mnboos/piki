import time
from pathlib import Path

import cv2
from ai_edge_litert.interpreter import Interpreter
import traceback
import numpy as np

# from PIL import Image
import multiprocessing as mp
from ctypes import c_float

prob_threshold = mp.Value(c_float, 0.5)

models = {"nanodet", "mobilenetv2_ssdlite", "mobilenetv3_ssdlite", ""}

try:
    print("Loading model...")
    model_path = str(
        Path(__file__).parent
        / "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite"
    )
    is_quant = "quant" in model_path.lower()

    ip = Interpreter(model_path=model_path)
    ip.allocate_tensors()
    inp_id = ip.get_input_details()[0]["index"]
    out_det = ip.get_output_details()
    out_id0 = out_det[0]["index"]
    out_id1 = out_det[1]["index"]
    out_id2 = out_det[2]["index"]
    out_id3 = out_det[3]["index"]

    image_classes = {
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

    def detect_objects(image: np.ndarray, out_size=(300, 300)):
        # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # orig_shape = image.shape
        # print("orig: ", orig_shape)

        image = cv2.resize(image, out_size, interpolation=cv2.INTER_AREA)
        # img_arr = np.array(resized_img)

        if not is_quant:
            image = image.astype(np.float32) / 128 - 1

        # ======================================================================
        # FIX: Add the batch dimension to make the input tensor 4D
        # Before: img_arr.shape was (300, 300, 3)
        # After:  input_tensor.shape will be (1, 300, 300, 3)
        # img_arr = np.expand_dims(img_arr, axis=0)
        # ======================================================================

        ms = lambda: int(round(time.time() * 1000))

        def print_coco_label(cl_id, t):
            print(
                "class: {}, label: {}, time: {:,} ms".format(
                    cl_id, image_classes[cl_id], t
                )
            )

        def print_output(res, original_img_size):
            boxes, classes, scores, num_det = res
            img_width, img_height = original_img_size

            i = 0

            n_obj = int(num_det[i])

            # print("{} - found objects:".format(fname))
            for j in range(n_obj):
                cl_id = int(classes[i][j]) + 1
                label = image_classes[cl_id]
                score = scores[i][j]
                if score < 0.5:
                    continue
                box = boxes[i][j]
                ymin, xmin, ymax, xmax = box

                # Calculate absolute coordinates
                abs_xmin = int(xmin * img_width)
                abs_ymin = int(ymin * img_height)
                abs_xmax = int(xmax * img_width)
                abs_ymax = int(ymax * img_height)

                print(
                    "  ", cl_id, label, score, [abs_xmin, abs_ymin, abs_xmax, abs_ymax]
                )

        t0 = time.perf_counter()
        ip.set_tensor(inp_id, [image])
        ip.invoke()
        tt = round((time.perf_counter() - t0) * 1000)
        print("Time:", tt, "ms")
        boxes = ip.get_tensor(out_id0)
        classes = ip.get_tensor(out_id1)
        scores = ip.get_tensor(out_id2)
        num_det = ip.get_tensor(out_id3)
        print_output((boxes, classes, scores, num_det), original_img_size=(0, 0))
        return []
except:
    traceback.print_exc()
    raise
