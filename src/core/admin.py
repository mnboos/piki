from django.contrib import admin

from .models import AimConfig, DetectionConfig, EventRecordingConfig, User, Video


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    pass


@admin.register(DetectionConfig)
class DetectionConfigAdmin(admin.ModelAdmin):
    pass


@admin.register(AimConfig)
class AimConfigAdmin(admin.ModelAdmin):
    pass


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    pass


@admin.register(EventRecordingConfig)
class EventRecordingConfigAdmin(admin.ModelAdmin):
    pass
