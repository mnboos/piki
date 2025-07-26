from django.apps import AppConfig
import atexit
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        # The `runserver` command runs this method twice. We use an environment
        # variable to ensure our setup code only runs in the main process.
        if os.environ.get("RUN_MAIN") != "true":
            return

        print("[DJANGO STARTUP] Initializing camera and AI workers...")

        # Import our shared objects and worker logic
        from .utils import stream
        # from picamera2 import Picamera2
        # from gpiozero import Servo

        # Initialize the shared multiprocessing objects
        # stream.frame_queue = mp.Queue(maxsize=1)
        # stream.results_queue = mp.Queue()
        # stream.stop_event = mp.Event()
        #
        # # Start the worker processes
        # for i in range(stream.NUM_AI_WORKERS):
        #     worker_proc = mp.Process(
        #         target=ai_worker_process,
        #         args=(
        #             stream.frame_queue,
        #             stream.results_queue,
        #             stream.stop_event,
        #             i,
        #         ),
        #         daemon=True,
        #     )
        #     stream.workers.append(worker_proc)
        #     worker_proc.start()

        # Define a cleanup function to run when Django exits
        def cleanup():
            print("[DJANGO SHUTDOWN] Stopping processes...")
            stream.executor.shutdown()
            if stream.camera:
                stream.camera.close()
            print("[DJANGO SHUTDOWN] Processes stopped.")

        #     if stream.stop_event:
        #         stream.stop_event.set()
        #     if stream.camera:
        #         stream.camera.stop_recording()
        #     # The daemon=True flag on processes is usually enough, but joining is good practice
        #     for worker in stream.workers:
        #         worker.join(timeout=1)
        #
        # # Register the cleanup function to be called on exit
        atexit.register(cleanup)
