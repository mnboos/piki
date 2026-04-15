from django.contrib import admin

from .models import AimConfig, DetectionConfig, User


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    pass


@admin.register(DetectionConfig)
class DetectionConfigAdmin(admin.ModelAdmin):
    pass


@admin.register(AimConfig)
class AimConfigAdmin(admin.ModelAdmin):
    pass
