#!/usr/bin/bash
set -e

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIBRATION_FILE="${SCRIPT_DIR}/calibration.yaml"
ROS_IMAGE_TOPIC="/camera/left/image_raw"

# ── Source environments ───────────────────────────────────────────────────────
# tros.b must be sourced BEFORE the venv, as it sets up ROS Python paths.
# If you source the venv first, Python picks up the venv interpreter and
# loses the tros.b site-packages.
source /opt/tros/humble/setup.bash

# ── Validate calibration file ─────────────────────────────────────────────────
if [ ! -f "$CALIBRATION_FILE" ]; then
    echo ""
    echo "  WARNING: calibration.yaml not found at $CALIBRATION_FILE"
    echo "  Stereo rectification will be disabled (need_rectify:=False)."
    echo "  Run camera calibration first for correct depth data."
    echo "  See: ros2 run camera_calibration cameracalibrator --help"
    echo ""
    NEED_RECTIFY="False"
    CALIBRATION_ARGS=""
else
    NEED_RECTIFY="True"
    CALIBRATION_ARGS="calibration_file_path:=${CALIBRATION_FILE}"
fi

# ── Start hobot_stereonet ─────────────────────────────────────────────────────
# This handles: MIPI camera → ISP → NV12 → stereo rectification → depth map
# It publishes rectified left/right images on /hbmem_img (shared memory, zero-copy)
echo "[piki] Starting hobot_stereonet..."
ros2 launch hobot_stereonet stereonet_model_web_visual_v2.2.launch.py \
    mipi_image_width:=1920 \
    mipi_image_height:=1080 \
    mipi_image_framerate:=30.0 \
    need_rectify:=${NEED_RECTIFY} \
    ${CALIBRATION_ARGS} &
STEREONET_PID=$!
echo "[piki] hobot_stereonet PID: $STEREONET_PID"

# Give the camera node a moment to initialise before Django tries to subscribe.
# hobot_stereonet takes a few seconds to load the BPU model and open the MIPI device.
echo "[piki] Waiting for stereonet to initialise..."
sleep 8

# ── Start Django ──────────────────────────────────────────────────────────────
echo "[piki] Starting Django..."
source "${SCRIPT_DIR}/.venv/bin/activate"
cd "${SCRIPT_DIR}/src"

export PYTHONUNBUFFERED=1
export MOCK_CAMERA_PATH=/dev/video0
export ROS_IMAGE_TOPIC="${ROS_IMAGE_TOPIC}"

python manage.py runserver --noreload 0.0.0.0:8000 &
DJANGO_PID=$!
echo "[piki] Django PID: $DJANGO_PID"

# ── Shutdown handler ──────────────────────────────────────────────────────────
# When you Ctrl+C, kill both processes cleanly.
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
# Block here until either process dies, then shut everything down.
wait -n $DJANGO_PID $STEREONET_PID
EXIT_CODE=$?
echo "[piki] A process exited (code: $EXIT_CODE), shutting down the other..."
cleanup
exit $EXIT_CODE