#!/usr/bin/bash
set -e

#fuser -k 8000/tcp

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# ── Source environments ───────────────────────────────────────────────────────
# tros.b must be sourced BEFORE the venv — it injects rclpy, hobot_dnn etc.
# into the Python path. Activating the venv afterwards layers on top correctly.
source /opt/tros/humble/setup.bash

# ── Pre-start cleanup ─────────────────────────────────────────────────────────
# Kill any stale ROS2 nodes from a previous crashed/unclean run.
# mipi_cam holds the MIPI hardware exclusively — if it's still alive from a
# prior run the new instance will fail with "rcl node's context is invalid".
echo "[piki] Cleaning up stale ROS2 nodes..."
pkill -x mipi_cam 2>/dev/null || true

# ── Fix tros.b runtime directories ───────────────────────────────────────────
# tros.b nodes write logs to /userdata/.roslog — create it if absent.
# nginx (websocket node) needs a logs/ dir relative to the working directory.
mkdir -p /userdata/.roslog
mkdir -p "${SCRIPT_DIR}/logs"
export ROS_LOG_DIR=/userdata/.roslog

# ── Start MIPI camera ─────────────────────────────────────────────────────────
# Publishes /image_left_raw and /image_right_raw.
# mipi_io_method:=shared_mem segfaults (HBM DMA driver bug in tros 2.5.2 — filed upstream).
# Using ros transport; ROS loaned-messages zero-copy is still active via FastDDS XML profile.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/tros/humble/lib/hobot_shm/config/shm_fastdds.xml
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
export ROS_DISABLE_LOANED_MESSAGES=0

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

# setsid puts the launch process in its own process group so that
# `kill -- -$CAM_PID` in cleanup() reaches all child nodes in one shot.
setsid ros2 launch mipi_cam mipi_cam_dual_channel.launch.py \
mipi_image_width:=1280 mipi_image_height:=640 mipi_lpwm_enable:=true mipi_image_framerate:=30.0 \
mipi_io_method:=ros \
mipi_camera_calibration_file_path:=/opt/tros/humble/lib/mipi_cam/config/SC230ai_dual_calibration.yaml &
CAM_PID=$!

sleep 2

# ── Start Django ──────────────────────────────────────────────────────────────
echo "[piki] Starting Django..."
source "${SCRIPT_DIR}/.venv/bin/activate"
cd "${SCRIPT_DIR}/src"

export PYTHONPATH=$PYTHONPATH:/opt/tros/humble/lib/python3.10/site-packages

export PYTHONUNBUFFERED=1

# /image_left_raw is the full-res left camera at 1280x640 NV12 (ISP output before
# stereonet downscales to its 640x352 depth model input). Use this for YOLO tiling:
# 1280x640 → up to 2 native 640x640 tiles → no resize → best YOLO accuracy.
ROS_IMAGE_TOPIC="/image_left_raw"
export ROS_IMAGE_TOPIC="${ROS_IMAGE_TOPIC}"

export  MODEL_FILE=/app/model/basic/yolov8_640x640_nv12.bin

python manage.py runserver --noreload 0.0.0.0:8000 &
DJANGO_PID=$!
echo "[piki] Django PID: $DJANGO_PID"

# ── Shutdown handler ──────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[piki] Shutting down..."
    kill $DJANGO_PID 2>/dev/null
    # Negative PID kills the entire process group started by setsid above,
    # ensuring mipi_cam and all other child nodes are terminated together.
    kill -- -$CAM_PID 2>/dev/null
    wait $DJANGO_PID 2>/dev/null
    wait $CAM_PID 2>/dev/null
    echo "[piki] Done."
}
trap cleanup SIGINT SIGTERM

# ── Wait ──────────────────────────────────────────────────────────────────────
# Unblocks if either process exits — then shut both down.
wait -n $DJANGO_PID $CAM_PID
EXIT_CODE=$?
echo "[piki] A process exited (code: $EXIT_CODE), shutting down..."
cleanup
exit $EXIT_CODE
