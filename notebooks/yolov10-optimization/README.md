# Convert and Optimize YOLOv10 with OpenVINO™

<p align="center">
    <img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/81ff3233-9c8d-4fe8-ab21-baf9ce530cff"/>
</p>

Real-time object detection aims to accurately predict object categories and positions in images with low latency. The YOLO series has been at the forefront of this research due to its balance between performance and efficiency. However, reliance on NMS and architectural inefficiencies have hindered optimal performance. YOLOv10 addresses these issues by introducing consistent dual assignments for NMS-free training and a holistic efficiency-accuracy driven model design strategy.

YOLOv10, built on the [Ultralytics Python package](https://pypi.org/project/ultralytics/) by researchers at [Tsinghua University](https://www.tsinghua.edu.cn/en/), introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

![yolov10-approach.png](https://github.com/ultralytics/ultralytics/assets/26833433/f9b1bec0-928e-41ce-a205-e12db3c4929a)

More details about model architecture you can find in original [repo](https://github.com/THU-MIG/yolov10), [paper](https://arxiv.org/abs/2405.14458) and [Ultralytics documentation](https://docs.ultralytics.com/models/yolov10/).

## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run and optimize PyTorch YOLO V10 with OpenVINO.

The tutorial consists of the following steps:

- Prepare PyTorch model
- Convert PyTorch model to OpenVINO IR
- Run model inference with OpenVINO
- Prepare and run optimization pipeline using NNCF
- Compare performance of the FP16 and quantized models.
- Run optimized model inference on video
- Launch interactive Gradio demo


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/yolov10-optimization/README.md" />
