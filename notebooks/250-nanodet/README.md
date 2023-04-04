# NanoDet model implementation with OpenVINO™

This notebook demonstrates how to convert and use [NanoDet](https://github.com/RangiLyu/nanodet) PyTorch model 
with OpenVINO.

<div style="text-align:center">
    <img src="https://raw.githubusercontent.com/sahilpmehra/images/main/nanodet_pytorch_result.png" width="70%"/>
</div>

NanoDet is a FCOS-style one-stage anchor-free object detection model which using Generalized Focal Loss as classification and regression loss.

Real-time object detection is often used as a key component in computer vision systems. Applications that use real-time object detection models include video analytics, robotics, autonomous vehicles, multi-object tracking and object counting, medical image analysis, and many others.


<div style="text-align:center">
    <img src="https://raw.githubusercontent.com/sahilpmehra/images/main/nanodet_arch.webp" width="70%"/>
</div>

More about the model and its details can be found in the following repository:
[NanoDet](https://github.com/RangiLyu/nanodet)

## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run PyTorch NanoDet model with OpenVINO.

The tutorial consists of the following steps:

* Prepare PyTorch model
* Validate original model
* Convert PyTorch model to ONNX
* Convert ONNX model to OpenVINO IR
* Validate converted model
* Download and prepare dataset
* Evaluating performance by AP (average Precision)


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md). 
All additional required libraries will be installed inside this notebook.
