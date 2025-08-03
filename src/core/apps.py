from django.apps import AppConfig
import os


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

        print("Camera setup", stream)

        # def cleanup():
        #     print("[DJANGO SHUTDOWN] Stopping processes...")
        #     stream.executor.shutdown()
        #     if stream.camera:
        #         stream.camera.close()
        #     print("[DJANGO SHUTDOWN] Processes stopped.")
        #
        # atexit.register(cleanup)
