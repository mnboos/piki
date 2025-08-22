## Servo setup

1. Install the dependencies
```bash
apt install pigpio python3-pigpio

uv add gpiozero pigpio
```

2. Start the gpio daemon
```bash
gpiod
```

3. Run the following python script

```python
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, AngularServo
from time import sleep

Device.pin_factory = PiGPIOFactory()

servo = AngularServo(12, min_angle=-170, max_angle=170)

while True:
    servo.min()
    sleep(1)
    servo.mid()
    sleep(1)
    servo.max()
    sleep(1)

```
## Setup Orange Pi 3B

- check: https://wiki.friendlyelec.com/wiki/index.php/NPU#Installing_RKNN_Runtime
```bash
# install
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2/rknpu2
sudo cp ./runtime/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./runtime/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/
sudo cp ./runtime/Linux/librknn_api/include/* /usr/include/

# check for v2
strings /usr/bin/rknn_server |grep 'build@'

# is this necessary ???
# fix broken link
cd ~/rknn-toolkit2/rknpu2/examples/3rdparty/mpp/Linux/aarch64
rm -f librockchip_mpp.so librockchip_mpp.so.1
ln -s librockchip_mpp.so.0 librockchip_mpp.so
ln -s librockchip_mpp.so.0 librockchip_mpp.so.1
```

Download the [armnndelegate](https://github.com/ARM-software/armnn/releases) for tflite (litert)

```bash
sudo apt install --no-install-recommends \
    git \
    build-essential \
    libcap-dev \
    clinfo \
    mesa-opencl-icd

cd /usr/lib/aarch64-linux-gnu/
sudo ln -s libOpenCL.so.1 libOpenCL.so

sudo usermod -a -G render $USER

echo 'export RUSTICL_ENABLE=panfrost' >> ~/.bashrc

# there should be no permission errors here now
clinfo

sudo reboot

cd src/piki
export MOCK_CAMERA_VIDEO_PATH=/home/dietpi/src/data/16701023-hd_1920_1080_60fps.mp4
LD_LIBRARY_PATH=/home/dietpi/src/data/arm ./run.sh
```

## Setup Radxa Zero 3W

**Links**
- https://radxa-repo.github.io/bullseye/
- RKNN installation: https://docs.radxa.com/en/zero/zero3/app-development/rknn_install
- RKNN examples: https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov5#2-current-support-platform
- Radxa headless setup: https://docs.radxa.com/en/template/sbc/radxa-os/headless#wireless

**Setup steps for Armbian**
1. flash image

```bash
sudo apt update -y

sudo apt install --no-install-recommends -y \
    git \
    build-essential \
    python3-dev \
    ffmpeg \
    v4l-utils \
    cmake \
    pkg-config

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

bash

cd ~
mkdir src
git clone https://github.com/mnboos/rknn-test.git
cd rknn-test
uv venv --system-site-packages
. .venv/bin/activate
uv sync

cp librknnrt.so /usr/lib/

# Download this: https://github.com/radxa-pkg/radxa-overlays/blob/main/arch/arm64/boot/dts/rockchip/overlays/rk3568-npu-enable.dts
wget https://raw.githubusercontent.com/radxa-pkg/radxa-overlays/refs/heads/main/arch/arm64/boot/dts/rockchip/overlays/rk3568-npu-enable.dts

sudo armbian-add-overlay rk3568-npu-enable.dts
sudo reboot

python run_resnet.py
```
**Compile Rockchip MPP**
```bash
cd ~
git clone https://github.com/rockchip-linux/mpp -b develop
mpp/build/linux/aarch64
./make-Makefiles.bash
sudo make install
sudo ldconfig
```

**Set fixed NPU frequency**
do this all with sudo
```bash
echo userspace > /sys/class/devfreq/fde40000.npu/governor
echo 900000000 > /sys/class/devfreq/fde40000.npu/userspace/set_freq

# verify current frequency
cat /sys/class/devfreq/fde40000.npu/cur_freq
```

**Setup steps for Radxa OS**
1. flash image
2. `sudo rsetup` and run update. *DO NOT* just `apt upgrade`
3. reboot
4. `sudo rsetup -> Overlays -> Yes -> Manage overlays -> Enable NPU` -> reboot
5. install cmake
6. install python 3.11
7. `pip install rknn-toolkit2`

**Setup**
```bash
apt update -y

apt install --no-install-recommends -y
    git \
    rknpu2-rk356x \
    libcap-dev \
    build-essential \
    v4l-utils

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Download and install rknn2
wget https://github.com/radxa-pkg/rknn2/releases/download/2.3.0-1/rknpu2-rk356x_2.3.0-1_arm64.deb
apt install ./rknpu2-rk356x_2.3.0-1_arm64.deb



```
Download and enable overlay
find the artifact here: 
    - https://github.com/radxa-pkg/radxa-overlays/actions/runs/16897851208/job/47870978948 (it's named something like: linux-rk356x.zip)
    or
    - https://github.com/Qengineering/Radxa-Zero-3-NPU-Ubuntu22
copy `rk3568-npu-enable.dtbo` to `/boot/dtb/rockchip/overlay/`
then modify /boot/dietpiEnv.txt: 

```
overlay_path=rockchip
# Multiple prefixes are supported separated by space
overlay_prefix=radxa-zero3 rk3568 rockchip
overlays=npu-enable
```


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

> Run `dietpi-config`, go into `Display Options` and activate RPi Camera and then reboot the device and try again.
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




































