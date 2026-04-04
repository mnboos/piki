from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0006_tracking_enabled"),
    ]

    operations = [
        migrations.AddField(
            model_name="detectionconfig",
            name="servo_smooth_factor",
            field=models.FloatField(default=1.0),
        ),
        migrations.AddField(
            model_name="detectionconfig",
            name="servo_dead_zone",
            field=models.FloatField(default=1.5),
        ),
    ]
