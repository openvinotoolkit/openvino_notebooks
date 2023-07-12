# Convert and Optimize YOLOv8 with OpenVINO™
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/230-yolov8-optimization/230-yolov8-optimization.ipynb)

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/212105105-f61c8aab-c1ff-40af-a33f-d0ed1fccc72e.png"/>
</p>

This tutorial explains how to convert and optimize the [YOLOv8](https://github.com/ultralytics/) PyTorch model with OpenVINO.


## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run and optimize PyTorch Yolo V8 with OpenVINO.

The tutorial consists of the following steps:
- Prepare the PyTorch model.
- Download and prepare the dataset.
- Validate the original model.
- Convert the PyTorch model to OpenVINO IR.
- Validate the converted model.
- Prepare and run NNCF post-training optimization pipeline.
- Compare accuracy of the FP32 and quantized models.
- Compare performance of the FP32 and quantized models.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
