from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    pass


class DetectionConfig(models.Model):
    """Singleton model for persisting detection/display tuning options."""

    show_boxes = models.BooleanField(default=True)
    show_mask = models.BooleanField(default=False)
    show_rois = models.BooleanField(default=False)
    show_seg = models.BooleanField(default=False)
    conf_threshold = models.FloatField(default=0.4)
    pixelcount_threshold = models.IntegerField(default=500)
    min_area = models.IntegerField(default=500)
    mog2_history = models.IntegerField(default=500)
    mog2_var_threshold = models.IntegerField(default=16)
    denoise_kernelsize = models.IntegerField(default=7)
    mask_transparency = models.FloatField(default=0.5)
    servo_pid_kp = models.FloatField(default=1.0)
    servo_pid_ki = models.FloatField(default=0.0)
    servo_pid_kd = models.FloatField(default=0.0)
    servo_dead_zone = models.FloatField(default=1.5)

    # --- Detection stability (Phase A: hysteresis, min-streak, EMA, ghost) ---
    # conf_threshold above acts as the "enter" threshold (lock entry, event trigger).
    # conf_threshold_keep is the lower hysteresis bound: a confirmed lock survives
    # while matched detections stay at or above this value.
    conf_threshold_keep = models.FloatField(default=0.25)
    # Same-class consecutive inference completions required before a detection
    # is treated as confirmed (counts down on misses).
    min_consecutive_frames = models.IntegerField(default=2)
    # EMA factor for the locked-target bbox: new = (1-α)·old + α·measured.
    # Lower = smoother (more lag); higher = more responsive (more jitter).
    bbox_ema_alpha = models.FloatField(default=0.4)
    # Window after the last real detection during which the previous detection
    # set is still drawn on the MJPEG stream for visual continuity.
    ghost_frames_ms = models.IntegerField(default=300)

    # --- SORT-style tracker (Phase B) ---
    # When enabled, on_done feeds detections through an IoU + Kalman tracker
    # that maintains identity across frames and fills brief detection gaps.
    tracker_enabled = models.BooleanField(default=True)
    tracker_iou_threshold = models.FloatField(default=0.3)
    tracker_max_misses = models.IntegerField(default=10)
    tracker_confirm_hits = models.IntegerField(default=3)

    class Meta:
        verbose_name = "Detection Config"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> "DetectionConfig":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class Video(models.Model):
    filename = models.CharField(max_length=255)
    file = models.FileField(upload_to="videos/")
    size_bytes = models.PositiveIntegerField(default=0)
    source = models.CharField(max_length=20, default="uploaded")
    created_at = models.DateTimeField(auto_now_add=True)

    # Per-event linkage for the technical-log feature.  ``event_id`` is the
    # timestamp+uuid stem shared with the JSONL sidecar; ``log_file`` is the
    # FileField pointing to that sidecar.  Blank for non-event sources.
    event_id = models.CharField(max_length=64, blank=True, default="")
    log_file = models.FileField(upload_to="videos/", blank=True, null=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.filename


class EventRecordingConfig(models.Model):
    """Singleton model for event-triggered recording settings."""

    enabled = models.BooleanField(default=False)
    pre_buffer_seconds = models.IntegerField(default=5)
    post_trigger_seconds = models.IntegerField(default=10)
    trigger_classes = models.JSONField(default=list)
    cooldown_seconds = models.IntegerField(default=30)

    class Meta:
        verbose_name = "Event Recording Config"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> "EventRecordingConfig":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class SplashConfig(models.Model):
    """Singleton model for splash (relay/solenoid) configuration."""

    enabled = models.BooleanField(default=False)
    trigger_classes = models.JSONField(default=list)
    delay_seconds = models.FloatField(default=0.5)
    duration_seconds = models.FloatField(default=1.0)
    cooldown_seconds = models.FloatField(default=10.0)

    class Meta:
        verbose_name = "Splash Config"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> "SplashConfig":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class ExclusionZone(models.Model):
    """Polygonal region in normalized [0, 1] camera-frame coordinates that must
    be ignored by detection, aiming, and recording triggers.  Rectangles are
    stored as 4-point polygons, so polygon support is a frontend-only addition
    later — no migration required.
    """

    name = models.CharField(max_length=64, default="zone")
    enabled = models.BooleanField(default=True)
    # JSON-encoded list of [x, y] pairs, each in [0, 1].  At least 3 points.
    points = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return self.name


class AimConfig(models.Model):
    """Singleton model holding servo aim configuration."""

    target_classes = models.JSONField(default=list)
    servo_enabled = models.BooleanField(default=False)
    target_lock_duration = models.FloatField(default=3.0)
    vertical_angle_offset = models.FloatField(default=0.0)
    pan_invert = models.BooleanField(default=False)
    tilt_invert = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Aim Config"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> "AimConfig":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj
