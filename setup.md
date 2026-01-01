# Setup Guide

Complete installation and configuration guide for the Video Analytics Benchmarking Platform across different hardware accelerators.

## Table of Contents

- [Prerequisites](#prerequisites)
- [CPU-Only Setup](#cpu-only-setup-no-acceleration)
- [NPU Setup (iMX8 Plus / iMX93)](#npu-setup)
- [Coral Edge TPU Setup](#coral-edge-tpu-setup)
- [Hailo-8 Setup](#hailo-8-setup)
- [Model Conversion](#model-conversion)


---

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04+, Debian 11+, or Yocto Linux
- **Python Version**: 3.9 (recommended) - 3.6 to 3.9 supported for Coral TPU
- **Git**: For cloning repositories
- **Root Access**: Required for some hardware accelerator installations

### General Setup

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install essential tools
sudo apt-get install -y git curl gnupg python3-pip

# Verify Python version
python3 --version  # Should be 3.9.x ideally
```

---

## CPU-Only Setup (No Acceleration)

### 1. Clone Repository

```bash
git clone https://bitbucket.org/andersdx/yolov5s_detection_cpu.git
cd yolov5s_detection_cpu
```

### 2. Python Version Check

If Python version is 3.11, downgrade to 3.9:
- Follow [Python 3.9 installation guide](https://www.rosehosting.com/blog/how-to-install-python-3-9-on-debian-11/)
- Remove existing symbolic link
- Create new symbolic link to Python 3.9

### 3. Install Dependencies

```bash
# Install ultralytics (YOLOv5)
pip3 install ultralytics

# Install project requirements
pip3 install -r requirements.txt
```

### 4. Run CPU Inference

**For ARM Architecture:**
```bash
python3 yolov5s_detection_cpu_arm.py \
    --weights yolov5s-int8-224.tflite \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --source video_test.mp4 \
    --imgsz 224 224 \
    --view-img
```

**For x86/Intel Architecture:**
```bash
python3 yolov5s_detection_cpu_x86.py \
    --weights yolov5s-int8-224.tflite \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --source video_test.mp4 \
    --imgsz 224 224 \
    --view-img
```

---

## NPU Setup

### iMX8 Plus / iMX93 Internal NPU

### 1. Verify NPU Library

**For iMX8 Plus:**
```bash
ls /usr/lib/libvx_delegate.so
```

**For iMX93:**
```bash
ls /usr/lib/libethosu_delegate.so
```

If the library file exists, NPU is available.

### 2. Test NPU Functionality (iMX93)

```bash
# Navigate to TensorFlow Lite examples
cd /usr/bin/tensorflow-lite-2.9.1/examples

# Test with CPU only
python3 label_image.py

# Test with NPU acceleration
python3 label_image.py -e /usr/lib/libethosu_delegate.so
```

Compare performance to verify NPU is working.

### 3. Run NPU Inference

**For iMX93:**

Update the ext-delegate path in the script to `/usr/lib/libethosu_delegate.so`

```bash
python3 yolov5s_detection_cpu_npu.py \
    --weights yolov5s-int8-224.tflite \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --source video_test.mp4 \
    --imgsz 224 224
```

---

## Coral Edge TPU Setup

**⚠️ Important: Use `sudo` for all package installations**

### 1. Clone Repository

```bash
git clone https://ktsang14@bitbucket.org/andersdx/yolov5s_detection_edgetpu.git
cd yolov5s_detection_edgetpu
```

### 2. System Update

```bash
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y git curl gnupg
```

### 3. Add Coral Repository

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update
```

**If you get "apt-key is deprecated" error**, use this alternative method:

```bash
wget -qO - https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo gpg --no-default-keyring \
    --keyring gnupg-ring:/etc/apt/trusted.gpg.d/google.gpg --import -

sudo rm /etc/apt/trusted.gpg.d/google.gpg~
sudo chmod 644 /etc/apt/trusted.gpg.d/google.gpg

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt-get update
```

Reference: [Coral Edge TPU Issue #777](https://github.com/google-coral/edgetpu/issues/777)

### 4. Python Version Setup

**⚠️ pycoral only supports Python 3.6 - 3.9**

If your Python version is 3.11 or higher, downgrade to 3.9:

1. Follow [Python 3.9 installation guide](https://www.rosehosting.com/blog/how-to-install-python-3-9-on-debian-11/)
2. Remove existing Python symbolic link
3. Create new symbolic link:

```bash
sudo ln -s /usr/local/bin/python3.9 /usr/bin/python

# Verify versions
python3 --version
python --version
```

If Python version is correct (3.6-3.9), skip to step 5.

### 5. Install Edge TPU Runtime

```bash
# Install Edge TPU driver and runtime
sudo apt-get install -y gasket-dkms libedgetpu1-std

# Install pycoral library
sudo python3 -m pip install \
    --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

# Install Python dependencies
sudo apt-get install -y python3 python3-pip
pip3 install --upgrade pip setuptools wheel
```

### 6. Install Python Packages

```bash
# Core packages
python3 -m pip install numpy
python3 -m pip install opencv-python

# If OpenCV fails, try headless version
python3 -m pip install opencv-python-headless

# Additional dependencies
python3 -m pip install tqdm pyyaml

# Ensure Edge TPU library is installed
sudo apt install libedgetpu1-std
```

### 7. Reboot System

```bash
sudo reboot now
```

### 8. Verify Coral TPU Detection

**Check if Coral is detected by system:**
```bash
lspci -nn | grep 089a
```

Expected output: `03:00.0 System peripheral: Device 1ac1:089a`

**Check if PCIe driver is loaded:**
```bash
ls /dev/apex_0
```

Expected output: `/dev/apex_0`

### 9. Run Edge TPU Inference

**For ARM Architecture:**
```bash
sudo python3 yolov5s_detection_edgetpu_arm.py \
    -m yolov5s-int8-224_edgetpu.tflite \
    --video video_test.mp4 \
    --conf_thresh 0.25 \
    --iou_thresh 0.45
```

**For Intel/x86 Architecture:**
```bash
python3 yolov5s_detection_edgetpu_x86.py \
    -m yolov5s-int8-224_edgetpu.tflite \
    --video video_test.mp4 \
    --conf_thresh 0.25 \
    --iou_thresh 0.45
```

---

## Hailo-8 Setup

### 1. Install Hailo PCIe Driver

```bash
sudo dpkg --install hailort-pcie-driver_4.15.0_all.deb
```

### 2. Install Docker

Follow standard Docker installation for your platform.

### 3. Start Hailo Docker Container

```bash
# Start new container
./run_tappas_docker.sh \
    --tappas-image hailo-docker-tappas-v3.26.0.tar \
    --container-name tappas_001

# Resume existing container
./run_tappas_docker.sh \
    --container-name tappas_001 \
    --resume
```

### 4. Clone Hailo Detection Repository

Inside the Docker container:

```bash
# Navigate to detection directory
cd /local/workspace/tappas/apps/h8/gstreamer/general/detection

# Clone the repository
git clone https://bitbucket.org/andersdx/yolov5s_detection_hailo.git
```

### 5. Add Custom Post-Processing Files

```bash
# Navigate to post-processing directory
cd /local/workspace/tappas/core/hailo/libs/postprocesses/detection/

# Copy custom YOLOv5 post-processing files
sudo cp /local/workspace/tappas/apps/h8/gstreamer/general/detection/yolov5s_detection_hailo/hailo_hef/yolo_hailortpp.cpp .

sudo cp /local/workspace/tappas/apps/h8/gstreamer/general/detection/yolov5s_detection_hailo/hailo_hef/yolo_hailortpp.hpp .
```

### 6. Regenerate Post-Processing Libraries

```bash
cd /local/workspace/tappas/scripts/gstreamer/
./install_hailo_gstreamer.sh
```

### 7. Configure Detection Application

```bash
cd /local/workspace/tappas/apps/h8/gstreamer/general/detection

# Edit detection script
nano detection.sh
```

Update the following paths in `detection.sh`:

```bash
readonly DEFAULT_HEF_PATH="$TAPPAS_WORKSPACE/apps/h8/gstreamer/general/detection/yolov5s_detection_hailo/hailo_hef/yolov5s.hef"

readonly DEFAULT_VIDEO_SOURCE="$TAPPAS_WORKSPACE/apps/h8/gstreamer/general/detection/yolov5s_detection_hailo/hailo_hef/video_test.mp4"

readonly DEFAULT_NETWORK_NAME="yolov5s"
```

### 8. Run Hailo Inference

```bash
./detection.sh --show-fps --print-device-stats
```

---

## Model Conversion

### Converting Models to Hailo Format (.hef)

**Workflow:**
```
.pt → ONNX/TensorFlow → .har → optimized.har → .hef
```

### Step 1: Convert PyTorch to ONNX

```bash
# Install ultralytics
pip install ultralytics

# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install requirements
pip install -r requirements.txt

# Export to ONNX (224x224 resolution)
python3 export.py \
    --img-size 224 224 \
    --weights yolov5s.pt \
    --include onnx
```

This creates `yolov5s.onnx` ready for Hailo Dataflow Compiler.

### Step 2: Setup Hailo Dataflow Compiler Environment

**Documentation:** [Hailo Dataflow Compiler Guide](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-26-0/?sp_referrer=install/install.html) (requires account)

**Create Virtual Environment:**
```bash
# Create virtualenv
virtualenv hailo_env

# Activate virtualenv
source hailo_env/bin/activate

# Install Hailo Dataflow Compiler
pip install hailo_dataflow_compiler-X.XX.X-py3-none-linux_x86_64.whl
```

**Resume Environment (for future sessions):**
```bash
source hailo_env/bin/activate
```

### Step 3: Parse ONNX to HAR

```bash
hailo parser onnx \
    --net-name yolov5s \
    --har-path /path/to/output/ \
    --hw-arch hailo8 \
    --parsing-report-path /path/to/report.txt \
    /path/to/yolov5s_224.onnx
```

**With specific output nodes:**
```bash
hailo parser onnx \
    --net-name yolov5s \
    --har-path /path/to/output/ \
    --hw-arch hailo8 \
    --parsing-report-path /path/to/report.txt \
    /path/to/yolov5s_224.onnx \
    --end-node-names "/model.24/m.2/Conv" "/model.24/m.1/Conv" "/model.24/m.0/Conv"
```

### Step 4: Optimize HAR

```bash
hailo optimize \
    /path/to/output/yolov5s.har \
    --hw-arch hailo8 \
    --use-random-calib-set \
    --model-script /path/to/model_script.txt
```

### Step 5: Compile to HEF

```bash
hailo compiler \
    /path/to/yolov5s_optimized.har \
    --hw-arch hailo8 \
    --output-dir /path/to/output/hailo_hef/ \
    --auto-model-script /path/to/auto_script/script.py
```

The final `.hef` file is now ready for deployment on Hailo-8 hardware.

---

## Troubleshooting

### Coral Edge TPU Issues

**Device Not Detected:**
```bash
# Check USB connection (for USB accelerator)
lsusb | grep "Global Unichip Corp"

# Check PCIe connection (for M.2/PCIe accelerator)
lspci -nn | grep 089a

# Reinstall driver
sudo apt-get remove libedgetpu1-std
sudo apt-get install libedgetpu1-std
sudo reboot
```

**Permission Denied:**
```bash
# Add user to plugdev group
sudo usermod -aG plugdev $USER

# Create udev rule
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a6e", GROUP="plugdev"' | \
    sudo tee /etc/udev/rules.d/99-edgetpu-accelerator.rules

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Log out and back in for group changes to take effect
```

**Python Version Incompatibility:**
```bash
# pycoral only supports Python 3.6 - 3.9
python3 --version

# If version is too high, follow Python 3.9 installation guide
# https://www.rosehosting.com/blog/how-to-install-python-3-9-on-debian-11/
```

### NPU Issues

**Library Not Found:**
```bash
# Verify NPU library exists
ls /usr/lib/libvx_delegate.so      # iMX8 Plus
ls /usr/lib/libethosu_delegate.so  # iMX93

# If missing, reinstall TensorFlow Lite runtime
pip3 install tflite-runtime
```

### Hailo-8 Issues

**Driver Not Loaded:**
```bash
# Check if Hailo device is detected
lspci | grep Hailo

# Verify driver installation
hailortcli fw-control identify

# Reinstall driver if needed
sudo dpkg --install hailort-pcie-driver_4.15.0_all.deb
sudo reboot
```

**Docker Container Issues:**
```bash
# List running containers
docker ps -a

# Remove old container
docker rm tappas_001

# Start fresh container
./run_tappas_docker.sh --tappas-image hailo-docker-tappas-v3.26.0.tar --container-name tappas_001
```

### General Issues

**OpenCV Installation Fails:**
```bash
# Try headless version
pip3 uninstall opencv-python
pip3 install opencv-python-headless
```

**Import Errors:**
```bash
# Verify all packages are installed
pip3 list | grep -E "numpy|opencv|pycoral|tqdm"

# Reinstall if needed
pip3 install --force-reinstall numpy opencv-python tqdm pyyaml
```

**Low FPS / Performance:**
- Verify hardware accelerator is actually being used
- Check thermal throttling: `cat /sys/class/thermal/thermal_zone*/temp`
- Monitor CPU usage: `htop`
- Reduce input resolution if needed
- Check system logs for errors: `dmesg | tail -50`

---

## Quick Reference

### Command Summary

**CPU (ARM):**
```bash
python3 yolov5s_detection_cpu_arm.py --weights yolov5s-int8-224.tflite --source video.mp4 --imgsz 224 224
```

**CPU (x86):**
```bash
python3 yolov5s_detection_cpu_x86.py --weights yolov5s-int8-224.tflite --source video.mp4 --imgsz 224 224
```

**Edge TPU (ARM):**
```bash
sudo python3 yolov5s_detection_edgetpu_arm.py -m yolov5s-int8-224_edgetpu.tflite --video video.mp4
```

**Edge TPU (x86):**
```bash
python3 yolov5s_detection_edgetpu_x86.py -m yolov5s-int8-224_edgetpu.tflite --video video.mp4
```

**Hailo-8:**
```bash
./detection.sh --show-fps --print-device-stats
```

---
