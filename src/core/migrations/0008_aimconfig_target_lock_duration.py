from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0007_servo_smoothing"),
    ]

    operations = [
        migrations.AddField(
            model_name="aimconfig",
            name="target_lock_duration",
            field=models.FloatField(default=3.0),
        ),
    ]
