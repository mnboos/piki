#!/usr/bin/bash
set -e

#fuser -k 8000/tcp

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# stereonet publishes the rectified left image on this topic (640x352, NV12, shared mem).
# Django subscribes to this — no separate mipi_cam process needed.
ROS_IMAGE_TOPIC="/hbmem_img"

# ── Source environments ───────────────────────────────────────────────────────
# tros.b must be sourced BEFORE the venv — it injects rclpy, hobot_dnn etc.
# into the Python path. Activating the venv afterwards layers on top correctly.
source /opt/tros/humble/setup.bash

# ── Fix tros.b runtime directories ───────────────────────────────────────────
# tros.b nodes write logs to /userdata/.roslog — create it if absent.
# nginx (websocket node) needs a logs/ dir relative to the working directory.
mkdir -p /userdata/.roslog
mkdir -p "${SCRIPT_DIR}/logs"
export ROS_LOG_DIR=/userdata/.roslog

# ── Start hobot_stereonet ─────────────────────────────────────────────────────
# Owns the MIPI hardware exclusively. Handles:
#   SC230AI sensors → ISP (noise reduction, WDR) → rectification → NV12
# Publishes:
#   /hbmem_img          — rectified left image, 640x352 NV12, zero-copy shared mem
#   /depth_map          — disparity/depth map from stereonet BPU model
#   /hobot_stereonet_visual — colourised depth visualisation (for debugging)
#
# NOTE: need_rectify:=False because the camera EEPROM already contains the
# calibration matrices (Kl, Kr, Dl, Dr, R, t) — stereonet loads them
# automatically and rectifies internally regardless of this flag.
# Set to True only if you provide an external calibration_file_path override.
echo "[piki] Starting hobot_stereonet..."
ros2 launch hobot_stereonet stereonet_model_v2.2.launch.py \
    mipi_image_width:=640 \
    mipi_image_height:=352 \
    mipi_lpwm_enable:=True \
    mipi_image_framerate:=30.0 \
    need_rectify:=False \
    io_method:=shared_mem &
STEREONET_PID=$!
echo "[piki] hobot_stereonet PID: $STEREONET_PID"

# Give stereonet time to load the BPU model and open the MIPI device before
# Django tries to subscribe to /hbmem_img.
echo "[piki] Waiting for stereonet to initialise..."
sleep 10

# ── Start Django ──────────────────────────────────────────────────────────────
echo "[piki] Starting Django..."
source "${SCRIPT_DIR}/.venv/bin/activate"
cd "${SCRIPT_DIR}/src"

export PYTHONPATH=$PYTHONPATH:/opt/tros/humble/lib/python3.10/site-packages

export PYTHONUNBUFFERED=1
# Tell the ROS node inside Django which topic to subscribe to.
export ROS_IMAGE_TOPIC="${ROS_IMAGE_TOPIC}"

python manage.py runserver --noreload 0.0.0.0:8000 &
DJANGO_PID=$!
echo "[piki] Django PID: $DJANGO_PID"

# ── Shutdown handler ──────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[piki] Shutting down..."
    kill $DJANGO_PID 2>/dev/null
    kill $STEREONET_PID 2>/dev/null
    wait $DJANGO_PID 2>/dev/null
    wait $STEREONET_PID 2>/dev/null
    echo "[piki] Done."
}
trap cleanup SIGINT SIGTERM

# ── Wait ──────────────────────────────────────────────────────────────────────
# Unblocks if either process exits — then shut both down.
wait -n $DJANGO_PID $STEREONET_PID
EXIT_CODE=$?
echo "[piki] A process exited (code: $EXIT_CODE), shutting down..."
cleanup
exit $EXIT_CODE