import time
import random

from django.apps import AppConfig
import os
import sys

class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        # The `runserver` command runs this method twice. We use an environment
        # variable to ensure our setup code only runs in the main process.
        if os.environ.get("RUN_MAIN"):


            # print("environ: ", os.environ, flush=True)
            # print("sys.argv: ", sys.argv, flush=True)

            print("[DJANGO STARTUP] Initializing camera and AI workers...")


            if len(sys.argv) >= 2 and sys.argv[1] == "runserver":

                # Import our shared objects and worker logic
                from .utils.metrics import create_queue_manager, retrieve_queue

                print("Starting queue manager...")

                create_queue_manager()
                time.sleep(1)
                manager_started = False
                retries = 0
                while not manager_started:
                    try:
                        manager_started = retrieve_queue()
                        print("Queue manager started.")
                    except EOFError:
                        backoff = (2 ** retries) * 0.1
                        print(f"Queue manager not yet started, retrying again in {backoff}...")
                        retries += 1
                        time.sleep(backoff)
                        continue


                from .utils.stream import start_stream_nonblocking
                start_stream_nonblocking()
                print("Start streaming...")



            # def cleanup():
            #     print("[DJANGO SHUTDOWN] Stopping processes...")
            #     stream.executor.shutdown()
            #     if stream.camera:
            #         stream.camera.close()
            #     print("[DJANGO SHUTDOWN] Processes stopped.")
            #
            # atexit.register(cleanup)
