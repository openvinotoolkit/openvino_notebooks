# PyTorch to ONNX and OpenVINO IR Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F306-pytorch-object-detection-accuracy-check-and-quantization%2F306-pytorch-object-detection-accuracy-check-and-quantization.ipynb)

![Ultralytics Yolov5 results](https://user-images.githubusercontent.com/44352144/177097174-cfe78939-e946-445e-9fce-d8897417ef8e.png)


This tutorial demonstrates step-by-step instructions to perform the model quantization by OpenVINO [Post-Training Optimization Tool (POT)](https://docs.openvino.ai/latest/pot_introduction.html), and to compare model accuracy between FP32 precision and quantized INT8 precision and to show a demo of running model inference based[Ultralytics Yolov5](https://github.com/ultralytics/yolov5) sample with OpenVINO 2022.1 backend.

## Notebook Contents

The notebook follow [Ultralytics Yolov5](https://github.com/ultralytics/yolov5) project to get Yolov5-m model with OpenVINO Intermediate Representation (IR) formats. Then use OpenVINO [Post-Training Optimization Tool (POT)](https://docs.openvino.ai/latest/pot_introduction.html) API to quantize model based on Ultralytics provided Non-Max Suppression (NMS) processing. And also compare accuracy drop between FP32 model and POT quantized INT8 by "DefaultQuantization" algorithm. Finally, refer Ultralytics provided Yolov5 sample "detect.py" to inference the INT8 model and check performance of model inference with OpenVINO sync API enabled.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
