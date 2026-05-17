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
            print(
                f"[{os.getpid()}]--- CUSTOM RELOADER: Change detected in  {filename} ---"
            )
            print(
                "--- CUSTOM RELOADER: I am the WATCHER process. I will now tell the worker to die. ---"
            )
            print("--- CUSTOM RELOADER: Waiting 2 seconds before proceeding... ---")
            time.sleep(2)  # You could add a delay or other logic here

            # 3. CRUCIAL: Call the original function to actually perform the reload.
            # If you forget this, your app will never reload.
            original_trigger_reload(filename)
            print(
                "--- CUSTOM RELOADER: Original reload trigger has !!been called. -ffhh--"
            )

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

    def _load_detection_config(self):
        """Load persisted DetectionConfig from DB into shared memory."""
        try:
            from .models import DetectionConfig  # noqa: PLC0415
            from .utils.shared import app_settings, mask_transparency, prob_threshold, servo_dead_zone, servo_pid_kd, servo_pid_ki, servo_pid_kp, settings  # noqa: PLC0415

            config = DetectionConfig.load()
            app_settings.debug_settings.show_boxes = config.show_boxes
            app_settings.debug_settings.show_mask = config.show_mask
            app_settings.debug_settings.show_rois = config.show_rois
            prob_threshold.value = config.conf_threshold
            settings.foreground_mask_options.pixelcount_threshold.value = config.pixelcount_threshold
            settings.foreground_mask_options.min_area.value = config.min_area
            settings.foreground_mask_options.mog2_history.value = config.mog2_history
            settings.foreground_mask_options.mog2_var_threshold.value = config.mog2_var_threshold
            settings.foreground_mask_options.denoise_kernelsize.value = config.denoise_kernelsize
            mask_transparency.value = config.mask_transparency
            servo_pid_kp.value = config.servo_pid_kp
            servo_pid_ki.value = config.servo_pid_ki
            servo_pid_kd.value = config.servo_pid_kd
            servo_dead_zone.value = config.servo_dead_zone
            print(
                f"[DJANGO STARTUP] Loaded detection config: show_boxes={config.show_boxes}, "
                f"show_mask={config.show_mask}, show_rois={config.show_rois}, "
                f"conf={config.conf_threshold}, mog2_history={config.mog2_history}",
                flush=True,
            )
        except Exception:
            import traceback  # noqa: PLC0415
            print("[DJANGO STARTUP] Could not load DetectionConfig — using defaults.", flush=True)
            traceback.print_exc()

    def _load_aim_config(self):
        """Load persisted AimConfig from DB into shared memory."""
        try:
            from .models import AimConfig  # noqa: PLC0415
            from .utils.shared import app_settings  # noqa: PLC0415

            config = AimConfig.load()
            app_settings.aim_settings.target_classes = config.target_classes
            app_settings.aim_settings.servo_enabled = config.servo_enabled
            app_settings.aim_settings.target_lock_duration = float(config.target_lock_duration)
            print(
                f"[DJANGO STARTUP] Loaded aim config: servo_enabled={config.servo_enabled}, "
                f"classes={config.target_classes}, target_lock_duration={config.target_lock_duration}s",
                flush=True,
            )
        except Exception:
            import traceback  # noqa: PLC0415
            print("[DJANGO STARTUP] Could not load AimConfig — using defaults.", flush=True)
            traceback.print_exc()

    def ready(self):
        # The `runserver` command runs this method twice. We use an environment
        # variable to ensure our setup code only runs in the main process.
        print("sys.argv: ", sys.argv, flush=True)
        is_running_main = os.environ.get("RUN_MAIN") or "--noreload" in sys.argv
        pid = os.getpid()
        ppid = os.getppid()
        print(f"[Django-{pid}, ppid={ppid}] RUN_MAIN:  ", is_running_main)

        if is_running_main:
            # Load persisted settings from DB into shared memory.
            self._load_detection_config()
            self._load_aim_config()

            # monkey_patch_reloader()

            # print("environ: ", os.environ, flush=True)

            print("[DJANGO STARTUP] Initializing camera and AI workers...")

            if len(sys.argv) >= 2 and sys.argv[1] == "runserver":
                import threading

                from .utils.metrics import queue_manager, retrieve_queue

                # queue_manager.start() forks a child process — must happen in the
                # main thread before any other threads are spawned to avoid deadlock.
                print("Starting queue manager.....")
                queue_manager.start()
                atexit.register(queue_manager.shutdown)

                def _start_background():
                    # stream.py has heavy ROS2 imports at module level — import it
                    # here in the background thread so it never blocks the main thread.
                    from .utils.stream import stream_nonblocking  # noqa: PLC0415

                    retrieve_queue()
                    print("Queue manager started.")
                    stream_nonblocking()
                    print("Start streaming...")

                threading.Thread(target=_start_background, daemon=True, name="piki-startup").start()

            # def cleanup():
            #     print("[DJANGO SHUTDOWN] Stopping processes...")
            #     stream.executor.shutdown()
            #     if stream.camera:
            #         stream.camera.close()
            #     print("[DJANGO SHUTDOWN] Processes stopped.")
            #
            # atexit.register(cleanup)
