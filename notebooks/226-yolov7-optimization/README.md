# Convert and Optimize YOLOv7 with OpenVINO™

[YOLOv7 results](https://raw.githubusercontent.com/WongKinYiu/yolov7/main/figure/horses_prediction.jpg)

This tutorial explains how to convert and optimize the [YOLOv7](https://github.com/WongKinYiu/yolov7) PyTorch model with OpenVNO.


## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run and optimize PyTorch Yolo V7 with OpenVINO.

The tutorial consists of the following steps:
- Prepare PyTorch model
- Download and prepare dataset
- Validate original model
- Convert PyTorch model to ONNX
- Convert ONNX model to OpenVINO IR
- Validate converted model
- Prepare and run NNCF Postraining optimization pipeline
- Compare accuracy of the FP32 and quantized models
- Compare performance of the FP32 and quantized models

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
