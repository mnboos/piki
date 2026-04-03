# Piki: An AI-Powered Cat Deterrent System

## 1. Management Summary

### The Problem
Neighborhood cats frequently enter our garden, using it as a litter box. This creates an unsanitary environment and poses a significant health risk, especially for babies and small children who play in the garden. Cat feces can transmit harmful parasites and bacteria, making a clean and safe outdoor space a top priority.

### Our Solution
**Piki** is an autonomous, humane, and cost-effective system designed to solve this problem. Using a **D-Robotics RDK X5** single-board computer and a 180° stereo fisheye camera, Piki employs real-time object detection to identify animals (or any user-selected YOLO class) as soon as they enter a monitored area. Upon detection, pan/tilt servos aim at the target and the system can trigger a harmless deterrent such as a brief water spray or ultrasonic tone. The whole system is managed through a simple web interface.

---

## 2. Technical Details

### Hardware Setup

**Core Components:**
- **Single-Board Computer:** [D-Robotics RDK X5](https://developer.d-robotics.cc/rdk_doc/en/) — Sunrise X5 SoC with a dedicated BPU (Brain Processing Unit) NPU for hardware-accelerated YOLOv8 inference.
- **Camera:** Stereo 180° fisheye camera connected to the RDK X5's MIPI CSI port. The `hobot_stereonet` ROS2 node rectifies and undistorts the fisheye images before they reach the detection pipeline.
- **Pan/Tilt Servos:** Two standard 50 Hz hobby servos (±90° range each) for aiming:
  - **Pan servo** → physical pin **32** (PWM6)
  - **Tilt servo** → physical pin **33** (PWM7)

  See [Section 3](#3-installation--setup) for wiring and enabling hardware PWM.
- **Power Supply:** 5 V for the SBC; servo power from a separate 5–6 V supply is recommended for heavier loads.
- **(Optional) Deterrent:** Relay-controlled water valve, ultrasonic speaker, etc.

### Software Architecture

| Layer | Technology |
|---|---|
| AI inference | YOLOv8 on Horizon BPU via `hobot_dnn` |
| Camera / depth | `hobot_stereonet` ROS2 node (tros.b, humble) |
| Backend | Django 5 + django-ninja REST API |
| Frontend | Vue 3 + Vite + TailwindCSS |
| Servo control | `Hobot.GPIO` hardware PWM (RPi.GPIO-compatible) |
| Persistence | SQLite (Django ORM) |

**Key Python dependencies:**
- `Hobot.GPIO` — GPIO/PWM control (pre-installed on RDK X5)
- `numpy`, `opencv-contrib-python-headless` — image processing
- `django`, `django-ninja` — web server and REST API
- `rknn-toolkit-lite2` — NPU model runtime (aarch64 only)
- `rich` — terminal logging

### Web Interface Features
- **Live camera feed** with bounding box / mask / ROI overlays
- **Servo Aiming** — enable/disable auto-aim and select which YOLO classes trigger the servos (any of the 80 COCO classes)
- **Manual Servo Debug** — 3×3 preset grid (top-left … bottom-right), pan/tilt sliders (−90° to +90°), and a Move button for calibration
- Confidence threshold, motion sensitivity, and min-area sliders
- Background reset button

### Angle Calculation
The stereonet node removes fisheye distortion. Angles are computed with the pinhole `atan2` model using calibrated intrinsics (`fx=fy≈258`, `cx≈314`, `cy≈160` for 640×352 output), giving an effective HFOV ≈ 102° and VFOV ≈ 68°.

```
pan_angle  = degrees(atan2(pixel_x − cx, fx))
tilt_angle = degrees(atan2(pixel_y − cy, fy))
```

Intrinsics can be overridden via env vars `CAMERA_FX`, `CAMERA_FY`, `CAMERA_CX`, `CAMERA_CY`, `CAMERA_FRAME_W`, `CAMERA_FRAME_H`.

---

## 3. Installation & Setup

### A. D-Robotics RDK X5 (primary target)

#### Enable Hardware PWM for Servos

Pins 32 and 33 support hardware PWM but require a device-tree overlay. This only needs to be done once:

```bash
# Enable the PWM3 overlay (pins 32 + 33 = PWM6 + PWM7)
echo -e 'dtoverlay=dtoverlay_pwm3\n' | sudo tee /boot/config.txt
sudo reboot
```

> **Note:** This overlay disables the I2C bus that shares those pins (`340c0000`). If you need that I2C bus, use a different PWM pair — see `/boot/overlays/README.txt` for the full list.

After rebooting, verify the PWM chip appears:
```bash
ls /sys/class/pwm/   # should show pwmchip0 and pwmchip1 (or similar)
```

#### Servo Wiring

| Servo wire | Connect to |
|---|---|
| Signal (yellow/white) | Physical pin **32** (pan) or **33** (tilt) |
| Power (red) | 5 V — physical pin 2 or 4 |
| Ground (black/brown) | GND — physical pin 6, 9, 14, 20, 25, … |

Override pins via env vars: `SERVO_PAN_PIN=32 SERVO_TILT_PIN=33` (physical/BOARD numbering).

#### Install Python Dependencies

```bash
uv sync
```

`Hobot.GPIO` is pre-installed system-wide on the RDK X5 image — no extra install needed.

#### Run

```bash
./run.sh
```

Access the web UI at `http://<board-ip>/`.

---

## 4. Usage: The Web Interface

### Starting the Application

```bash
cd /path/to/piki
./run.sh
```

### Accessing the GUI

Open `http://<board-ip>/` in any browser on the same network.

### Controls

| Section | What it does |
|---|---|
| **Mode** | Switch between Boxes, Mask, and ROI visualisation |
| **Confidence** | YOLOv8 detection confidence threshold |
| **Motion sensitivity / Min area** | Background-subtraction filter tuning |
| **Reset background** | Clear the motion-detection baseline |
| **Servo Aiming → Enable** | Toggle auto-aim on/off |
| **Servo Aiming → Classes** | Pick which detected YOLO classes move the servos |
| **Manual Servo Debug** | Preset grid + sliders to aim servos manually for calibration |
