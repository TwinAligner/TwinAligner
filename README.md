# TwinAligner: Visual-Dynamic Alignment Empowers Physics-aware Real2Sim2Real for Robotic Manipulation

<div align="center">
<p>
  <a href="https://hwfan.io/about-me/">Hongwei Fan<sup>*</sup></a>,
  <a href="https://daihangpku.github.io/">Hang Dai<sup>*</sup></a>,
  <a href="https://jiyao06.github.io/">Jiyao Zhang<sup>*</sup></a>,
  <a href="https://kingchou007.github.io/">Jinzhou Li</a>,
  <a href="https://qiyangyan.github.io/web/">Qiyang Yan</a>,
</p>
</div>
<div align="center">
<p>
  <a href="https://github.com/ZhaoYujie2002">Yujie Zhao</a>,
  <a href="https://github.com/GasaiYU">Mingju Gao</a>,
  <a href="https://github.com/Happy-Boat">Jinghang Wu</a>,
  <a href="https://ha0tang.github.io/">Hao Tang</a>,
  <a href="https://zsdonghao.github.io/">Hao Dong</a>
</p>
</div>
<div align="center">
(* indicates equal contribution)
</div>
<div align="center">
<a href="https://arxiv.org/abs/2512.19390">
<img src="https://img.shields.io/badge/arXiv-2512.19390-b31b1b" alt="arXiv">
</a>
<a href="https://twin-aligner.github.io/">
<img src="https://img.shields.io/badge/Project_Page-TwinAligner-green" alt="Project Page">
</a>
</div>

<p align="center">
  <img src="assets/method.png" alt="Teaser" style="center" />
</p>

This repository contains the official implementation of [TwinAligner: Visual-Dynamic Alignment Empowers Physics-aware Real2Sim2Real for Robotic Manipulation](https://twin-aligner.github.io/).

## Installation

### Environment

- Tested on a workstation with an NVIDIA RTX 4090 (24GB) GPU and 64 GB RAM.
- Compatible with Ubuntu 20.04 / 22.04, CUDA 12.1, and GCC 11.4.

```bash
git clone --branch opensource --recurse-submodules https://github.com/TwinAligner/TwinAligner.git

conda create -n twinaligner -y python=3.10
conda activate twinaligner
conda install -c conda-forge -c nvidia/label/cuda-12.1.0 -y ffmpeg colmap=3.9.0 eigen=3.4.0 aria2 pybind11 boost-cpp nvidia/label/cuda-12.1.0::cuda-toolkit
pip install uv
bash pipelines/installation/0_download_ckpts.sh
bash pipelines/installation/1_install_env.sh
bash pipelines/installation/2_install_foundationpose.sh
```

## Visual Real2Sim

### Download Example Assets

The example assets and videos can be used directly in the following steps.

```bash
## Assets
bash pipelines/installation/0_download_assets.sh
## Videos
bash pipelines/installation/0_download_videos.sh
```

### Rigid Object

```bash
## 1. Perform Real2Sim
bash pipelines/object_real2sim.sh -i examples/milkbox -p box --steps 12345
## 2. Assetize. Please modify the timestamp generated in step 1.
bash pipelines/object_assetize.sh -i examples/milkbox -p box -t <your-timestamp> --steps 12
```

### Background / Robot

```bash
## 1. Perform Real2Sim
bash pipelines/background_real2sim.sh -i examples/background --steps 12
## 2. Assetize. Please modify the timestamp generated in step 1.
bash pipelines/background_assetize.sh -i examples/background -t <your-timestamp> --steps 1234
```

### View Alignment

#### Data Collection

Please refer to [twinaligner_traj_recorder](https://github.com/TwinAligner/twinaligner_traj_recorder).

To download the example data:

```bash
gdown 1a4rYbdEV-K2Kg4frTCXilfH-iEpd1QwR -O examples/
unzip examples/franka-track.zip -d examples/franka-track
```

#### Run Alignment

```bash 
# Step 1: Initialize camera pose using FoundationPose.
# NOTE: Please segment the robot when the SAM3 GUI (Gradio) interface appears.
bash pipelines/foundation_pose_robot.sh -i examples/franka-track -t asset --steps 12
# Step 2: Perform PSO optimization.
# NOTE: Please segment the table when the SAM3 GUI (Gradio) interface appears.
bash pipelines/refine_cam_pose.sh -i examples/franka-track -t asset --steps 12
# Step 3: Copy the best parameters.
cp examples/franka-track/pso_logs/best_w2c.txt assets/realsense/cam_extr.txt
cp examples/franka-track/cam_K.txt assets/realsense/cam_K.txt
```

## Dynamic Real2Sim

Coming soon.

## Collecting Demonstrations

### Preparing the Environment

Default teleoperation uses keyboard:

```bash
bash pipelines/installation/3_install_curobo.sh
# NOTE: Please place your license file under `checkpoints/anygrasp_license`.
bash pipelines/installation/4_install_anygrasp.sh
bash pipelines/installation/5_install_ros.sh
```

Teleoperation via [PICO VR](https://xr-robotics.github.io/) is also supported:

```bash
bash pipelines/installation/6_install_xrobotoolkit.sh
```

### Teleoperation

- Step 1: Start ROSCore.

```bash
bash pipelines/start_roscore.sh
```

- Step 2: Launch the teleoperation system.

Default teleoperation uses keyboard:

```bash
# Move end effector in the robot axes:
#   W/S - forward / backward (X)
#   A/D - left / right (Y)
#   Q/E - up / down (Z)
#   R/F - roll +/-
#   T/G - pitch +/-
#   Y/H - yaw +/-
#   I/O - close / open gripper
# Recording: 1 - recollect & reset, 2 - start record, 3 - end record, 4 - save & reset
bash pipelines/start_teleop_daemon.sh --tracker_type keyboard
```

For PICO VR:

1. Start the XRoboToolkit app in the PICO headset.
2. Run the following daemon.

```bash
bash pipelines/start_teleop_daemon.sh --tracker_type pico
```

### Automatic Pick-and-Place

```bash
bash pipelines/start_auto_collect_daemon.sh
```

### Rendering Collected States with 3DGS

```bash
bash pipelines/render_data.sh
```

### Open-source Plan

- [x] Visual Reconstruction
- [x] Viewpoint Alignment
- [ ] Dynamic Real2Sim
- [x] Keyboard Teleoperation
- [x] VR Teleoperation
- [x] Automatic Pick-and-Place
- [x] Rendering with 3DGS
- [ ] More Documents
- [ ] More Demos

## Acknowledgements

If you find TwinAligner useful, please consider citing:

```
@article{fan2025twinaligner,
author    = {Hongwei Fan and Hang Dai and Jiyao Zhang and Jinzhou Li and Qiyang Yan and Yujie Zhao and Mingju Gao and Jinghang Wu and Hao Tang and Hao Dong},
title     = {TwinAligner: Visual-Dynamic Alignment Empowers Physics-aware Real2Sim2Real for Robotic Manipulation},
year={2025},
eprint={2512.19390},
archivePrefix={arXiv},
primaryClass={cs.RO},
url={https://arxiv.org/abs/2512.19390},
}
```
