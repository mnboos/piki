from django.urls import path

from . import views
from .api import api

urlpatterns = [
    path("", views.index, name="index"),
    path("move_servo/", views.move_servo, name="move_servo"),
    # path("upload_video/", views.upload_video, name="upload_video"),
    path("config_ai/", views.config_ai, name="config_ai"),
    path("config/", views.config, name="config"),
    path("api/", api.urls, name="api"),
]
