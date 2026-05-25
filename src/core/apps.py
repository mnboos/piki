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
            from .utils.shared import (  # noqa: PLC0415
                app_settings,
                bbox_ema_alpha,
                min_consecutive_frames,
                prob_threshold,
                prob_threshold_keep,
                servo_dead_zone,
                servo_kalman_meas_noise,
                servo_kalman_process_noise,
                servo_pid_kd,
                servo_pid_ki,
                servo_pid_kp,
                settings,
                tracker_confirm_hits,
                tracker_delta_t,
                tracker_enabled,
                tracker_inertia,
                tracker_iou_threshold,
                tracker_max_misses,
            )

            config = DetectionConfig.load()
            app_settings.debug_settings.show_boxes = config.show_boxes
            prob_threshold.value = config.conf_threshold
            prob_threshold_keep.value = min(float(config.conf_threshold_keep), float(config.conf_threshold))
            min_consecutive_frames.value = max(1, int(config.min_consecutive_frames))
            bbox_ema_alpha.value = max(0.0, min(1.0, float(config.bbox_ema_alpha)))
            tracker_enabled.value = 1 if config.tracker_enabled else 0
            tracker_iou_threshold.value = max(0.0, min(1.0, float(config.tracker_iou_threshold)))
            tracker_max_misses.value = max(0, int(config.tracker_max_misses))
            tracker_confirm_hits.value = max(1, int(config.tracker_confirm_hits))
            tracker_delta_t.value = max(1, int(config.tracker_delta_t))
            tracker_inertia.value = max(0.0, min(1.0, float(config.tracker_inertia)))
            settings.foreground_mask_options.pixelcount_threshold.value = config.pixelcount_threshold
            settings.foreground_mask_options.min_area.value = config.min_area
            settings.foreground_mask_options.mog2_history.value = config.mog2_history
            settings.foreground_mask_options.mog2_var_threshold.value = config.mog2_var_threshold
            settings.foreground_mask_options.denoise_kernelsize.value = config.denoise_kernelsize
            servo_pid_kp.value = config.servo_pid_kp
            servo_pid_ki.value = config.servo_pid_ki
            servo_pid_kd.value = config.servo_pid_kd
            servo_dead_zone.value = config.servo_dead_zone
            servo_kalman_process_noise.value = max(0.01, float(config.servo_kalman_process_noise))
            servo_kalman_meas_noise.value = max(0.01, float(config.servo_kalman_meas_noise))
            print(
                f"[DJANGO STARTUP] Loaded detection config: show_boxes={config.show_boxes}, "
                f"conf_enter={config.conf_threshold}, conf_keep={prob_threshold_keep.value}, "
                f"min_streak={min_consecutive_frames.value}, ema_alpha={bbox_ema_alpha.value}, "
                f"mog2_history={config.mog2_history}, "
                f"tracker={'on' if tracker_enabled.value else 'off'} "
                f"(iou={tracker_iou_threshold.value}, max_misses={tracker_max_misses.value}, "
                f"confirm_hits={tracker_confirm_hits.value}, "
                f"delta_t={tracker_delta_t.value}, inertia={tracker_inertia.value:.2f}), "
                f"kalman(proc={servo_kalman_process_noise.value}, "
                f"meas={servo_kalman_meas_noise.value})",
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
            from .utils.shared import (  # noqa: PLC0415
                app_settings,
                servo_aim_confidence,
                servo_pan_invert,
                servo_tilt_invert,
                vertical_angle_offset,
            )

            config = AimConfig.load()
            app_settings.aim_settings.target_classes = config.target_classes
            app_settings.aim_settings.servo_enabled = config.servo_enabled
            app_settings.aim_settings.target_lock_duration = float(config.target_lock_duration)
            app_settings.aim_settings.aim_confidence = float(config.aim_confidence)
            servo_aim_confidence.value = float(config.aim_confidence)
            vertical_angle_offset.value = float(config.vertical_angle_offset)
            app_settings.aim_settings.pan_invert = bool(config.pan_invert)
            app_settings.aim_settings.tilt_invert = bool(config.tilt_invert)
            servo_pan_invert.value = 1 if config.pan_invert else 0
            servo_tilt_invert.value = 1 if config.tilt_invert else 0
            print(
                f"[DJANGO STARTUP] Loaded aim config: servo_enabled={config.servo_enabled}, "
                f"classes={config.target_classes}, target_lock_duration={config.target_lock_duration}s, "
                f"aim_confidence={config.aim_confidence}, "
                f"vertical_angle_offset={config.vertical_angle_offset}°, "
                f"pan_invert={config.pan_invert}, tilt_invert={config.tilt_invert}",
                flush=True,
            )
        except Exception:
            import traceback  # noqa: PLC0415
            print("[DJANGO STARTUP] Could not load AimConfig — using defaults.", flush=True)
            traceback.print_exc()

    def _load_event_recording_config(self):
        """Load persisted EventRecordingConfig from DB into shared memory."""
        try:
            from .models import EventRecordingConfig  # noqa: PLC0415
            from .utils.shared import (  # noqa: PLC0415
                event_cooldown_seconds,
                event_post_trigger_seconds,
                event_pre_buffer_seconds,
                event_recording_enabled,
                event_trigger_classes,
                event_trigger_classes_lock,
            )

            config = EventRecordingConfig.load()
            if config.enabled:
                event_recording_enabled.set()
            event_pre_buffer_seconds.value = float(config.pre_buffer_seconds)
            event_post_trigger_seconds.value = float(config.post_trigger_seconds)
            event_cooldown_seconds.value = float(config.cooldown_seconds)
            with event_trigger_classes_lock:
                event_trigger_classes.clear()
                event_trigger_classes.extend([c.lower() for c in config.trigger_classes])
            print(
                f"[DJANGO STARTUP] Loaded event recording config: enabled={config.enabled}, "
                f"pre={config.pre_buffer_seconds}s, post={config.post_trigger_seconds}s, "
                f"cooldown={config.cooldown_seconds}s, classes={config.trigger_classes}",
                flush=True,
            )
        except Exception:
            import traceback  # noqa: PLC0415
            print("[DJANGO STARTUP] Could not load EventRecordingConfig — using defaults.", flush=True)
            traceback.print_exc()

    def _load_splash_config(self):
        """Load persisted SplashConfig from DB into shared memory."""
        try:
            from .models import SplashConfig  # noqa: PLC0415
            from .utils.shared import (  # noqa: PLC0415
                pump_duty,
                splash_cooldown,
                splash_delay,
                splash_duration,
                splash_enabled,
            )

            config = SplashConfig.load()
            if config.enabled:
                splash_enabled.set()
            else:
                splash_enabled.clear()
            splash_delay.value = float(config.delay_seconds)
            splash_duration.value = float(config.duration_seconds)
            splash_cooldown.value = float(config.cooldown_seconds)
            pump_duty.value = float(config.pump_duty)
            print(
                f"[DJANGO STARTUP] Loaded splash config: enabled={config.enabled}, "
                f"delay={config.delay_seconds}s, duration={config.duration_seconds}s, "
                f"cooldown={config.cooldown_seconds}s, pump_duty={config.pump_duty}%",
                flush=True,
            )
        except Exception:
            import traceback  # noqa: PLC0415
            print("[DJANGO STARTUP] Could not load SplashConfig — using defaults.", flush=True)
            traceback.print_exc()

    def ready(self):
        # The `runserver` command runs this method twice. We use an environment
        # variable to ensure our setup code only runs in the main process.
        print("sys.argv: ", sys.argv, flush=True)
        # Only run heavy startup when actually serving the app, not for
        # migrate/shell/test/collectstatic/etc. Two serving entrypoints:
        #   - manage.py runserver (dev)
        #   - daphne piki.asgi:application (prod, behind Caddy)
        cmd = sys.argv[1] if len(sys.argv) >= 2 else ""
        executable = os.path.basename(sys.argv[0]) if sys.argv else ""
        is_runserver = cmd == "runserver"
        is_daphne = executable == "daphne"
        if not (is_runserver or is_daphne):
            return

        # runserver's autoreloader runs ready() in both parent and child; only
        # the child (--noreload or RUN_MAIN=true) should do heavy startup.
        # daphne has no reloader, so it's always the worker.
        is_worker = is_daphne or "--noreload" in sys.argv or os.environ.get("RUN_MAIN") == "true"
        if not is_worker:
            return

        # Load persisted settings from DB into shared memory.
        self._load_detection_config()
        self._load_aim_config()
        self._load_event_recording_config()
        self._load_splash_config()

        # monkey_patch_reloader()

        # print("environ: ", os.environ, flush=True)

        print("[DJANGO STARTUP] Initializing camera and AI workers...")

        if is_runserver or is_daphne:
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
