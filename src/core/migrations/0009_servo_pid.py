from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0008_aimconfig_target_lock_duration"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="detectionconfig",
            name="servo_smooth_factor",
        ),
        migrations.AddField(
            model_name="detectionconfig",
            name="servo_pid_kp",
            field=models.FloatField(default=1.0),
        ),
        migrations.AddField(
            model_name="detectionconfig",
            name="servo_pid_ki",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="detectionconfig",
            name="servo_pid_kd",
            field=models.FloatField(default=0.0),
        ),
    ]
