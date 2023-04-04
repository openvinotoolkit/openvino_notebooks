# Convert and Optimize Temporal Shift Module (TSM) with OpenVINO™

This tutorial explains how to convert and optimize the [TSM](https://github.com/mit-han-lab/temporal-shift-module) PyTorch model with OpenVINO.


## Notebook Contents

This tutorial provides step-by-step instructions on how to run and optimize PyTorch TSM with OpenVINO.

The tutorial consists of the following steps:
- Prepare and load PyTorch TSM model
- Convert PyTorch model to ONNX
- Convert ONNX model to OpenVINO IR
- Download and prepare dataset
- Compare accuracy of PyTorch, ONNX, and OpenVINO IR models
- Optimize the OpenVINO IR model using post-training 8-bit integer quantization
- Compare accuracy and performance of the FP32 and quantized models


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
