import atexit
import os
import sys
import time

from django.apps import AppConfig


def monkey_patch_reloader():
    """
    Applies a monkey-patch to Django's autoreload.trigger_reload function.
    This code ONLY runs in the watcher process.
    """
    try:
        from django.utils import autoreload

        print("Monkey-patching Django's auto-reloader...")

        # 1. Store a reference to the original function
        original_trigger_reload = autoreload.trigger_reload

        def custom_trigger_reload(filename):
            # from .utils.shared import DJANGO_RELOAD_ISSUED, DJANGO_RELOAD_SEMAPHORE

            # DJANGO_RELOAD_ISSUED.set()

            from .utils.stream import reloading

            reloading.clear()

            # 2. Add your custom logic here
            print(f"[{os.getpid()}]--- CUSTOM RELOADER: Change detected in  {filename} ---")
            print("--- CUSTOM RELOADER: I am the WATCHER process. I will now tell the worker to die. ---")
            print("--- CUSTOM RELOADER: Waiting 2 seconds before proceeding... ---")
            time.sleep(2)  # You could add a delay or other logic here

            # 3. CRUCIAL: Call the original function to actually perform the reload.
            # If you forget this, your app will never reload.
            original_trigger_reload(filename)
            print("--- CUSTOM RELOADER: Original reload trigger has !!been called. -ffhh--")

            reloading.set()
            # DJANGO_RELOAD_ISSUED.clear()

        # 4. Replace the original function with your custom one
        autoreload.trigger_reload = custom_trigger_reload
        print("✅ Auto-reloader patched successfully.")

    except ImportError:
        # Handle cases where autoreload might not be available
        print("Could not import autoreload to apply patch.")


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        # The `runserver` command runs this method twice. We use an environment
        # variable to ensure our setup code only runs in the main process.
        print("sys.argv: ", sys.argv, flush=True)
        is_running_main = os.environ.get("RUN_MAIN") or "--noreload" in sys.argv
        pid = os.getpid()
        ppid = os.getppid()
        print(f"[Django-{pid}, ppid={ppid}] RUN_MAIN:  ", is_running_main)

        if is_running_main:
            # monkey_patch_reloader()

            # print("environ: ", os.environ, flush=True)

            print("[DJANGO STARTUP] Initializing camera and AI workers...")

            if len(sys.argv) >= 2 and sys.argv[1] == "runserver":
                # Import our shared objects and worker logic
                from .utils.metrics import queue_manager, retrieve_queue

                print("Starting queue manager.....")

                queue_manager.start()

                atexit.register(queue_manager.shutdown)

                time.sleep(1)
                manager_started = False
                retries = 0
                while not manager_started:
                    try:
                        manager_started = retrieve_queue()
                        print("Queue manager started.")
                    except EOFError:
                        backoff = (2**retries) * 0.1
                        print(f"Queue manager not yet started, retrying again in {backoff}...")
                        retries += 1
                        time.sleep(backoff)
                        continue

                from .utils.stream import stream_nonblocking

                stream_nonblocking()
                print("Start streaming...")

            # def cleanup():
            #     print("[DJANGO SHUTDOWN] Stopping processes...")
            #     stream.executor.shutdown()
            #     if stream.camera:
            #         stream.camera.close()
            #     print("[DJANGO SHUTDOWN] Processes stopped.")
            #
            # atexit.register(cleanup)
