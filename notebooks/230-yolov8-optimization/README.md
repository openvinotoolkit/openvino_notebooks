# Convert and Optimize YOLOv8 with OpenVINO™

![YOLOv8 results](https://user-images.githubusercontent.com/29454499/212105105-f61c8aab-c1ff-40af-a33f-d0ed1fccc72e.png)

This tutorial explains how to convert and optimize the [YOLOv8](https://github.com/ultralytics/) PyTorch\* model with OpenVNO.


## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run and optimize PyTorch\* Yolo V8 with OpenVINO.

The tutorial consists of the following steps:
- Prepare PyTorch model
- Download and prepare dataset
- Validate original model
- Convert PyTorch model to OpenVINO IR
- Validate converted model
- Prepare and run NNCF Postraining optimization pipeline
- Compare accuracy of the FP32 and quantized models
- Compare performance of the FP32 and quantized models

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).