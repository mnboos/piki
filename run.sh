#!/usr/bin/bash

. .venv/bin/activate
cd src

export PYTHONUNBUFFERED=1
python manage.py runserver --noreload 0.0.0.0:8000
