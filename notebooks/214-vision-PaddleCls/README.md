# PaddlePaddle Image Classification with OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F214-vision-PaddleCls%2F214-vision-PaddleCls.ipynb)


This demo shows how to run MobileNetV3 Large PaddePaddle model on OpenVINO natively. Instead of exporting the PaddlePaddle model to ONNX and then create the Intermediate Representation (IR) format through OpenVINO optimizer, we can now read direct from the Paddle Model without any conversions. It also covers the OpenVINO 2022.1 new feature guide:

* Preprocessing API
* Directly Loading a PaddlePaddle Model
* Auto-pluging
* AsyncInferQueue PythonAPI
* Performance Hint
  * LATENCY Mode
  * THROUGHPUT Mode
  
## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
