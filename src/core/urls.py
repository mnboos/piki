from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("video_feed/", views.video_feed, name="video_feed"),
    path("mask_feed/", views.mask_feed, name="mask_feed"),
    path("move_servo/", views.move_servo, name="move_servo"),
    # path("upload_video/", views.upload_video, name="upload_video"),
    path("config_ai/", views.config_ai, name="config_ai"),
    path("config/", views.config, name="config"),
]
