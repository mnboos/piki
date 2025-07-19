## Install `Picamera2`

See: [rpicam-apps(-lite)](https://www.raspberrypi.com/documentation/computers/camera_software.html#install-libcamera-and-rpicam-apps)

**Note:** Make sure to use the system python3 and not a python managed by uv, as there will be problems by uv-managed-python not detecting libcamera.

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
libcamera-vid --awbgain 1,1 -t 0 --inline --listen -o tcp://0.0.0.0:8888
```

**View stream**
Using VLC or simmilar open `tcp/h264://<ip>:8888` 

## Streaming with `ffmpeg`

**Stream**
```bash
picam-vid --nopreview --awbgain 1,1 -t 0 --codec yuv420 --width 1024 --height 768 --framerate 30 -o - | ffmpeg -f rawvideo -pix_fmt yuv420p -s 1024x768 -r 30 -i - -c:v h264_v4l2m2m -b:v 2000k -f mpegts -fflags flush_packets -preset ultrafast -tune zerolatency udp://192.168.1.148:8888
```

**Play**
```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -vf setpts=0 udp://192.168.1.149:8888
```
