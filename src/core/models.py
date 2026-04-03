from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    pass


class AimConfig(models.Model):
    """Singleton model holding servo aim configuration."""

    target_classes = models.JSONField(default=list)
    servo_enabled = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Aim Config"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> "AimConfig":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj
