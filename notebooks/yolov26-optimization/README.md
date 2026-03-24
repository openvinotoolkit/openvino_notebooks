# Convert and Optimize YOLO26 with OpenVINO™

<p align="center">
    <img src="https://github.com/user-attachments/assets/25b9a208-3ea6-4918-b75d-d8c6f9bfdd78"/>
</p>

[YOLO26](https://docs.ultralytics.com/models/yolo26/) is the latest Ultralytics release (January 2026), engineered from the ground up for edge and low-power devices. It is the recommended model for production workloads, replacing both YOLO11 and YOLO12.

YOLO stands for "You Only Look Once", a popular family of real-time object detection algorithms. YOLO26 introduces several key innovations over previous versions:

- **End-to-End NMS-Free Inference** — native end-to-end model producing predictions directly without non-maximum suppression (NMS), reducing latency and simplifying deployment.
- **Dual-Head Architecture** — one-to-one head (default, no NMS, max 300 detections) and one-to-many head (legacy, requires NMS, 8400 detections).
- **DFL Removal** — Distribution Focal Loss module removed, simplifying export and broadening edge hardware compatibility.
- **MuSGD Optimizer** — hybrid of SGD and Muon (inspired by Moonshot AI's Kimi K2) for more stable training and faster convergence.
- **ProgLoss + STAL** — improved loss functions with notable improvements in small-object detection.
- **Up to 43% Faster CPU Inference** — specifically optimized for edge computing environments.
- **Instance Segmentation Enhancements** — semantic segmentation loss and multi-scale proto module for superior mask quality.
- **Precision Pose Estimation** — Residual Log-Likelihood Estimation (RLE) for more accurate keypoint localization.
- **Refined OBB Decoding** — specialized angle loss for improved oriented bounding box detection accuracy.
- **YOLOE-26** — open-vocabulary variant supporting text prompts, visual prompts, and prompt-free zero-shot inference.

YOLO26 supports detection, instance segmentation, classification, pose estimation, oriented object detection (OBB), and open-vocabulary detection (YOLOE-26). More details can be found in the [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/).

This tutorial explains how to convert and optimize the YOLO26 PyTorch models with OpenVINO:

- [yolov26-object-detection](./yolov26-object-detection.ipynb) — object detection [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov26-optimization/yolov26-object-detection.ipynb)
- [yolov26-instance-segmentation](./yolov26-instance-segmentation.ipynb) — instance segmentation [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov26-optimization/yolov26-instance-segmentation.ipynb)
- [yolov26-keypoint-detection](./yolov26-keypoint-detection.ipynb) — keypoint/pose detection [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov26-optimization/yolov26-keypoint-detection.ipynb)
- [yolov26-obb](./yolov26-obb.ipynb) — oriented bounding boxes (OBB) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov26-optimization/yolov26-obb.ipynb)
- [yoloe-26-open-vocabulary](./yoloe-26-open-vocabulary.ipynb) — open-vocabulary detection and segmentation (YOLOE-26) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov26-optimization/yoloe-26-open-vocabulary.ipynb)

The tutorial consists of the following steps:
- Prepare the PyTorch model.
- Convert the PyTorch model to OpenVINO IR.
- Validate the converted model.
- Prepare and run NNCF post-training optimization pipeline.
- Compare performance of the FP16 and quantized models.
- Live demo with video inference.


## Installation Instructions

This is a self-contained example that relies solely on its own code.<br/>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/yolov26-optimization/README.md" />
