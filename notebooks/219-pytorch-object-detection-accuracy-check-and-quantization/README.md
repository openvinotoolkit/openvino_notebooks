# Quantize the Ultralytics YOLOv5 model and check accuracy using the OpenVINO POT API

![Ultralytics Yolov5 results](https://user-images.githubusercontent.com/44352144/177097174-cfe78939-e946-445e-9fce-d8897417ef8e.png)


This tutorial demonstrates step-by-step how to perform model quantization using the OpenVINO [Post-Training Optimization Tool (POT)](https://docs.openvino.ai/latest/pot_introduction.html), compare model accuracy between the FP32 precision and quantized INT8 precision models and run a demo of model inference based on sample code from [Ultralytics Yolov5](https://github.com/ultralytics/yolov5) with the OpenVINO backend.

## Notebook Contents

The notebook follow [Ultralytics Yolov5](https://github.com/ultralytics/yolov5) project to get Yolov5-m model with OpenVINO Intermediate Representation (IR) formats. Then use OpenVINO [Post-Training Optimization Tool (POT)](https://docs.openvino.ai/latest/pot_introduction.html) API to quantize model based on Ultralytics provided Non-Max Suppression (NMS) processing. And also compare accuracy drop between FP32 model and POT quantized INT8 by "DefaultQuantization" algorithm. Finally, refer Ultralytics provided Yolov5 sample "detect.py" to inference the INT8 model and check performance of model inference with OpenVINO sync API enabled.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
