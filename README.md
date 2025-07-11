## Install `Picamera2`

See: [rpicam-apps(-lite)](https://www.raspberrypi.com/documentation/computers/camera_software.html#install-libcamera-and-rpicam-apps)

```bash
apt update -y
apt install -y python3-picamera2 --no-install-recommends

apt install -y \
  build-essential \
  libcap-dev \
  rpicam-apps-lite \
  libcamera-dev \
  cmake \
  python3-libcamera

uv venv --system-site-packages
uv add picamera2
```

Then [modify `/boot/firmware/config.txt`](https://www.waveshare.com/wiki/RPi_Camera_(H)): 
```
camera_auto_detect=0
dtoverlay=ov5647
```

Then

> Run dietpi-config, go into Display Options and activate RPi Camera and then reboot the device and try again.
Source: https://dietpi.com/forum/t/want-to-connect-camera-module-3-to-raspberry-pi-3/19973


## RTSP Streaming

**Start stream**
```bash
libcamera-vid -t 0 --inline --listen -o tcp://0.0.0.0:8888
```

**View stream**
Using VLC or simmilar open `tcp/h264://<ip>:8888` 
