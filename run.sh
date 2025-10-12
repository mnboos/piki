#!/usr/bin/bash

. .venv/bin/activate
cd src

export PYTHONUNBUFFERED=1
export MOCK_CAMERA_PATH=/dev/video0
python manage.py runserver --noreload 0.0.0.0:8000
