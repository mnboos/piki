from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0027_detectionconfig_tracker_reid_enabled_and_more"),
    ]

    operations = [
        migrations.RemoveField(model_name="detectionconfig", name="show_mask"),
        migrations.RemoveField(model_name="detectionconfig", name="show_rois"),
        migrations.RemoveField(model_name="detectionconfig", name="mask_transparency"),
        migrations.RemoveField(model_name="detectionconfig", name="ghost_frames_ms"),
    ]
