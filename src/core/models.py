from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    pass


class DetectionConfig(models.Model):
    """Singleton model for persisting detection/display tuning options."""

    mode = models.CharField(max_length=20, default="boxes")
    conf_threshold = models.FloatField(default=0.4)
    pixelcount_threshold = models.IntegerField(default=500)
    min_area = models.IntegerField(default=500)
    mog2_history = models.IntegerField(default=500)
    mog2_var_threshold = models.IntegerField(default=16)
    denoise_kernelsize = models.IntegerField(default=7)
    mask_transparency = models.FloatField(default=0.5)
    tracker_type = models.CharField(max_length=8, default="CSRT")
    tracker_lost_threshold = models.IntegerField(default=5)
    tracking_enabled = models.BooleanField(default=True)
    servo_pid_kp = models.FloatField(default=1.0)
    servo_pid_ki = models.FloatField(default=0.0)
    servo_pid_kd = models.FloatField(default=0.0)
    servo_dead_zone = models.FloatField(default=1.5)

    class Meta:
        verbose_name = "Detection Config"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> "DetectionConfig":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class AimConfig(models.Model):
    """Singleton model holding servo aim configuration."""

    target_classes = models.JSONField(default=list)
    servo_enabled = models.BooleanField(default=False)
    target_lock_duration = models.FloatField(default=3.0)

    class Meta:
        verbose_name = "Aim Config"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> "AimConfig":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj
