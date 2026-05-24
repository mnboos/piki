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
| Backend | Django 5 + django-ninja REST API + Channels (WebSocket) |
| Video streaming | WebRTC via `aiortc`, hardware H.264 from `hobot_vio.libsrcampy` VPU encoder |
| Tracking | Norfair (IoU + Kalman, optional reid by color histogram) |
| Frontend | Vue 3 + Vite + PrimeVue + canvas overlay |
| Servo control | `Hobot.GPIO` hardware PWM (RPi.GPIO-compatible) |
| Persistence | SQLite (Django ORM) |

**Key Python dependencies:**
- `Hobot.GPIO` — GPIO/PWM control (pre-installed on RDK X5)
- `hobot_vio.libsrcampy`, `hobot_dnn` — VPU encoder + BPU runtime (system Python on `/opt/tros/humble`)
- `aiortc`, `av` — WebRTC peer connection + PyAV (we monkey-patch `aiortc`'s H.264 encoder to passthrough VPU output; pinned versions)
- `channels`, `daphne` — ASGI + WebSocket
- `norfair` — multi-object tracker
- `numpy`, `opencv-contrib-python-headless` — image processing
- `django`, `django-ninja` — web server and REST API
- `rich` — terminal logging

### Video Streaming Architecture

The live camera feed uses **WebRTC with hardware-accelerated H.264**, not MJPEG.
This gives roughly 10× the bandwidth efficiency and near-zero CPU cost on the
device. Overlays (detection boxes, servo crosshair, exclusion zones) are sent
as JSON over a separate WebSocket and rendered client-side on a `<canvas>`,
which keeps the encoder's path zero-copy.

#### Data flow

```
ROS /image_left_raw (NV12)
        │
        ▼
PikiVisionNode.listener_callback_hbm    ──►   process_frame()
        │                                          │
        │                                          ├──► motion detect + YOLO infer (BPU)
        │                                          │            │
        │                                          │            ▼
        │                                          │      Norfair tracker
        │                                          │            │
        │                                          │            ▼
        │                                          │      events.publish("detections", …)
        │                                          │            │
        │                                          ▼            ▼
        │                            HwH264Encoder           Django Channels
        │                            (Horizon VPU)           "events" group
        │                                  │                       │
        │                                  ▼                       ▼
        │                           annex-B NALs          ws://…/ws/events
        │                                  │                       │
        │                                  ▼                       │
        │                           per-peer asyncio.Queue (cap 4) │
        │                                  │                       │
        │                                  ▼                       │
        │                          HwH264Track.recv()              │
        │                                  │                       │
        │                                  ▼                       │
        │                  monkey-patched aiortc H264Encoder       │
        │                  → RTP packetize (STAP-A / FU-A)         │
        │                                  │                       │
        │                                  ▼                       │
        │                          DTLS-SRTP over UDP              │
        │                                  │                       │
        ▼                                  ▼                       ▼
            ┌──────────────────────────────────────────────────────┐
            │   browser                                            │
            │   <video srcObject=stream>   +   <canvas> overlay    │
            │      ▲                            ▲                  │
            │      └── WebRTC track             └── useEventStream │
            └──────────────────────────────────────────────────────┘
```

#### Why WebRTC (not MJPEG)

|                              | MJPEG (old)                   | WebRTC + hardware H.264 (current) |
|---|---|---|
| Bandwidth at 720p30          | ~30 Mbps                      | ~2–8 Mbps                          |
| Encode CPU                   | high (libjpeg per frame)      | near-zero (VPU does it)            |
| Motion compression           | none (every frame is full)    | inter-frame (P-frames are tiny)    |
| Poor-network adaptivity      | none                          | jitter buffer + NACK + PLI         |
| Server-side overlays         | required                      | impossible (zero-copy preserved)   |
| Setup complexity             | trivial                       | NAT/firewall sensitive             |

#### Hardware H.264 encoder

The Horizon SDK exposes a VPU encoder as `hobot_vio.libsrcampy.Encoder`. Our
wrapper in `src/core/utils/hw_encoder.py`:

1. Calls `Encoder.encode(channel, type=1, w, h)` to open the encoder. The
   Sunrise VPU requires 16-pixel-aligned dimensions, so a 1920×1080 input
   becomes 1920×1088 — Y and UV planes get copied into a pre-allocated padded
   buffer once per frame.
2. Pushes one NV12 frame via `send_frame`, pulls the encoded annex-B bitstream
   via `get_frame`. Bitrate is fixed at the encoder's default of **8000 kbps**
   with a default GOP — the Python binding does not expose
   `h264_cbr_params.bit_rate` or GOP knobs (the C-level API does; reaching
   them would require a small C extension or switching to a `Camera`+VPS
   pipeline. See the `TODO` in `hw_encoder.py`).
3. Splits the annex-B stream into individual NAL units (no start codes), and
   re-injects cached SPS/PPS in front of every IDR.

#### Why SPS/PPS re-injection matters

H.264 is a sequence of **NAL units** with type tags. The ones that matter
here:

- **SPS** (type 7) — Sequence Parameter Set. Describes the stream's resolution
  and profile. ~10–20 bytes.
- **PPS** (type 8) — Picture Parameter Set. Per-picture encoding details.
  ~4 bytes.
- **IDR** (type 5) — a fully self-contained keyframe. ~5–50 KB. Browsers can
  start decoding from any IDR.
- **P-frame** (type 1) — a tiny delta against previous frames.

A decoder cannot render anything without first seeing SPS + PPS. The Horizon
encoder emits them **once** at startup and never again, so a viewer that
joins the stream mid-flight — including any reconnect after a tab refresh or
network blip — would otherwise stare at a black frame until the encoder is
restarted. Our wrapper caches the first SPS and PPS it sees and prepends
them to every IDR (≈ once per second at the default GOP). Result: any new
peer is decoding within one keyframe interval.

#### The aiortc passthrough patch

`aiortc` was designed assuming you hand it `av.VideoFrame` instances
containing raw pixels, and it encodes them via libx264 (software, CPU-only).
That would defeat the entire point of having a hardware VPU — libx264 at
720p30 burns ~1–2 ARM cores.

We don't let it run. In `src/core/utils/webrtc.py` we replace
`aiortc.codecs.h264.H264Encoder.encode()` at module import with a one-liner
that says "the frame is already encoded — just RTP-packetize the NALs
attached to it and emit them."

Mechanics:

- `HwH264Track.recv()` returns a `_HwH264Frame` (a tiny `av.VideoFrame`
  subclass — needed because the C-extension `av.VideoFrame` has no
  `__dict__`). The subclass carries `_hw_nals` (the NAL list) and
  `pts`/`time_base` (so aiortc can compute the RTP 90 kHz timestamp
  correctly).
- The patched `encode()` reads `_hw_nals`, calls aiortc's existing
  `_packetize()` (which knows STAP-A and FU-A RTP fragmentation per
  RFC 6184), and returns the RTP payloads. aiortc's own libx264 path is
  never entered — verified by `enc.codec is None` after a round-trip.
- A startup assertion (`assert_passthrough_active()`) checks that the patch
  is still bound. If a future aiortc upgrade silently changes the method, the
  process refuses to start instead of falling back to libx264 unnoticed.

Trade-off: monkey-patching is hacky, but it is the standard way to inject
hardware encoding into aiortc on Jetson/Pi/custom-SoC projects. The
assertion catches dependency drift, and `aiortc` and `av` are version-pinned
in `pyproject.toml` for the same reason.

#### Detections, servo state, exclusion zones (the sidecar)

Per the user-side decision when this was designed, **all overlays are sent
as JSON over the existing `/ws/events` WebSocket** (Django Channels) and
drawn client-side. This preserves the zero-copy NV12 → H.264 → wire path —
the encoder ingests the raw camera buffer untouched. The frontend draws on a
DPR-aware `<canvas>` overlaid on the `<video>` element.

WS topics added/extended:

- `detections` — pushed from `on_done()` after each tracker update.
  Payload: `{ frame_ts_ns, detections: [{ tid, label, score, bbox: [xmin, ymin, xmax, ymax] }] }`
  with normalized 0–1 bboxes (axis order is xyxy on the wire so the frontend
  can draw without remembering the internal `[ymin, xmin, ymax, xmax]`).
- `tracker_status` — extended with
  `servo: { pan, tilt, kalman_pan, kalman_tilt }`. Publish throttle bumped
  to 10 Hz so the crosshair feels smooth.
- Exclusion zones come through the existing CRUD API + `ExclusionZoneOverlay.vue`.

Implications:

- Detection rendering lags the video by ~30–100 ms (WS hop vs WebRTC jitter
  buffer). Acceptable for slow targets like the cats this system is built
  for. A v2 could carry `frame_ts_ns` and align via
  `video.requestVideoFrameCallback()` if needed.
- The legacy server-side **mask** and **ROI** debug overlays are gone —
  they required raw mask data that isn't streamed. The toggle buttons were
  removed from the UI. Adding them back would require a separate binary
  channel (probably another WebSocket).

#### Signaling

`POST /api/webrtc/offer` is a stateless WHEP-style endpoint:

1. Browser creates `RTCPeerConnection`, `addTransceiver("video",
   {direction:"recvonly"})`, creates an offer, waits for ICE gathering to
   complete, POSTs `{sdp, type:"offer"}`.
2. Server (`webrtc.handle_offer` in `core/utils/webrtc.py`) builds a fresh
   `RTCPeerConnection`, attaches a `HwH264Track` whose queue is registered
   in the fanout set, sets the remote description, generates an answer, and
   returns `{sdp, type:"answer"}`.
3. ICE gathering on both sides happens before the SDP swap — no trickle ICE
   over a separate channel, which keeps the protocol stateless. One HTTP
   round-trip per peer, no long-lived signaling socket.

When the peer connection closes (any of `failed`/`closed`/`disconnected`),
the per-peer queue is removed from the fanout set and the `webrtc_active`
event is cleared if no peers remain. That gates the encoder in
`process_frame()` so the VPU is idle when nobody's watching.

#### Networking and ICE

WebRTC needs **both ends to be able to send UDP packets to each other**, not
just HTTP. Two common gotchas:

1. **Chrome's mDNS host-candidate obfuscation.** Modern Chrome replaces
   local IPs in host candidates with `xyz.local` names that aiortc cannot
   resolve. Fix: configure a STUN server on the browser side (we do — see
   `CameraFeed.vue`) so the browser also gathers `srflx` candidates the
   server can pair with.
2. **Asymmetric routing across VPNs.** If your laptop reaches the device
   through Tailscale/Wireguard/etc., but the device itself isn't on that
   VPN, the server has no candidate the browser can return packets to. ICE
   fails with `success=0 fail=N` even though plain HTTP works.

The recommended fix is **install Tailscale (or whatever VPN you use) on the
device itself**:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo systemctl restart piki
```

`aiortc` auto-discovers the `tailscale0` interface on next start and
advertises a `100.x` candidate that any other Tailscale device can reach
directly. Alternatives are running a TURN relay (e.g. `coturn` on a VPS) or
just accessing the device from its own LAN.

#### Operational notes

- **No "force IDR" hook.** `libsrcampy.Encoder` doesn't expose one from
  Python, so we rely on the encoder's default GOP (~1 keyframe/sec). A tab
  reload sees a usable picture within one GOP interval.
- **Single channel, single resolution.** Bitrate isn't exposed by the
  Python binding (TODO above). All peers share one VPU encode pipeline; the
  output is fanned out as RTP per peer, so adding more viewers does not
  multiply VPU cost. It does multiply network egress.
- **Debugging traffic.** `chrome://webrtc-internals` shows live
  `bytesReceived/s`, `framesPerSecond`, `keyFramesDecoded`, `pliCount`,
  codec, and current bitrate. Run `sudo iftop -i eth0` (or wlan0) on the
  device to confirm wire bytes match what the browser reports.
- **Verifying the hardware path.** If `top` on the device shows a Python
  thread at ~80% CPU while a peer is connected, the monkey-patch slipped
  and aiortc is software-encoding via libx264 — the startup assertion
  should have caught that, but it's worth a glance during operation. With
  the patch active, the encode load lives in the kernel/VPU driver and
  isn't visible to userspace process accounting.

### Web Interface Features
- **Live camera feed** (WebRTC, hardware H.264) with client-side detection
  bounding boxes, servo crosshair, and exclusion-zone overlays
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
| **Boxes toggle** | Show/hide client-side detection bounding boxes drawn on the canvas overlay |
| **Zones toggle** | Enter exclusion-zone editing mode (Detection tab) |
| **Confidence** | YOLOv8 detection confidence threshold |
| **Motion sensitivity / Min area** | Background-subtraction filter tuning |
| **Reset background** | Clear the motion-detection baseline |
| **Servo Aiming → Enable** | Toggle auto-aim on/off |
| **Servo Aiming → Classes** | Pick which detected YOLO classes move the servos |
| **Manual Servo Debug** | Preset grid + sliders to aim servos manually for calibration |
