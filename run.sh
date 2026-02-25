#!/usr/bin/bash
set -e

#fuser -k 8000/tcp

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


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
# CRITICAL for Shared Memory (HBM) to work with Django
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/tros/humble/lib/hobot_shm/config/shm_fastdds.xml
export RMW_FASTRTPS_USE_QOS_FROM_XML=1

## ── 2. Start MIPI Camera (Shared Memory Mode) ────────────────────────────────
#echo "[piki] Starting mipi_cam (Stereo Capture)..."
## We capture at 1280x704 because the ISP needs this for 2x 640x352 eyes
#ros2 launch mipi_cam mipi_cam.launch.py \
#    mipi_image_width:=1280 \
#    mipi_image_height:=704 \
#    mipi_video_device:=vps_camera \
#    mipi_io_method:=shared_mem \
#    mipi_out_format:=nv12 &
#CAM_PID=$!
#
#sleep 3 # Give ISP time to initialize
#
## ── 3. Start StereoNet Model ──────────────────────────────────────────────────
#echo "[piki] Starting hobot_stereonet (AI Engine)..."
## We use the core model launch and point it to the camera's raw output
#ros2 launch hobot_stereonet stereonet_model.launch.py \
#    stereo_image_topic:=/image_raw \
#    io_method:=shared_mem \
#    pub_rectified_hbm:=True &
#STEREONET_PID=$!

# ── 2. Start MIPI Camera ──────────────────────────────────────────────────────
#ros2 launch mipi_cam mipi_cam.launch.py \
#    mipi_image_width:=1280 \
#    mipi_image_height:=704 \
#    mipi_video_device:=vps_camera \
#    mipi_io_method:=shared_mem \
#    mipi_out_format:=nv12 \
#    mipi_out_topic:=/image_combine_raw &
#CAM_PID=$!
#ros2 launch mipi_cam mipi_cam_dual_channel.launch.py \
#    mipi_image_width:=1280 \
#    mipi_image_height:=704 \
#    mipi_io_method:=shared_mem \
#    mipi_out_format:=nv12 &
#CAM_PID=$!
#
#sleep 3
#
## ── 3. Start StereoNet Model ──────────────────────────────────────────────────
#ros2 launch hobot_stereonet stereonet_model.launch.py \
#    stereo_image_topic:=/image_combine_raw \
#    camera_info_topic:=/image_right_raw/camera_info &
#STEREONET_PID=$!
#

ros2 launch hobot_stereonet stereonet_model_no_web.launch.py \
mipi_image_width:=640 mipi_image_height:=352 mipi_lpwm_enable:=True mipi_image_framerate:=30.0 \
need_rectify:=False height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0 \
uncertainty_th:=0.1

sleep 5

# ── Start Django ──────────────────────────────────────────────────────────────
echo "[piki] Starting Django..."
source "${SCRIPT_DIR}/.venv/bin/activate"
cd "${SCRIPT_DIR}/src"

export PYTHONPATH=$PYTHONPATH:/opt/tros/humble/lib/python3.10/site-packages

export PYTHONUNBUFFERED=1

# stereonet publishes the rectified left image on this topic (640x352, NV12, shared mem).
# Django subscribes to this — no separate mipi_cam process needed.
ROS_IMAGE_TOPIC="/StereoNetNode/stereonet_visual"
export ROS_IMAGE_TOPIC="${ROS_IMAGE_TOPIC}"

export  MODEL_FILE=/app/model/basic/yolov5s_v7_640x640_nv12.bin

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