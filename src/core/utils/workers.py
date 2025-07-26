import multiprocessing as mp
import numpy as np
import cv2
import time
from queue import Empty


def run_ai_on_frame(image_np: np.ndarray, worker_id: int) -> str:
    """Symbolic AI function."""
    time.sleep(0.1 + (worker_id * 0.1))
    result_text = f"Worker {worker_id} at {time.strftime('%H:%M:%S')}"
    return result_text


def ai_worker_process(
    frame_queue: mp.Queue, results_queue: mp.Queue, stop_event: mp.Event, worker_id: int
) -> None:
    """The target function for each AI worker process."""
    print(f"[WORKER {worker_id}] Process started.")
    while not stop_event.is_set():
        try:
            frame_timestamp, frame_data = frame_queue.get(timeout=1)
        except Empty:
            continue
        except KeyboardInterrupt:
            break

        image_np = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        ai_result = run_ai_on_frame(image_np, worker_id)
        results_queue.put((frame_timestamp, worker_id, ai_result))

    print(f"[WORKER {worker_id}] Process stopped.")
