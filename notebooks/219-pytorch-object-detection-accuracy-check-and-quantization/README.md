# Quantize the Ultralytics YOLOv5 model and check accuracy using the OpenVINO POT API

![Ultralytics Yolov5 results](https://user-images.githubusercontent.com/44352144/177097174-cfe78939-e946-445e-9fce-d8897417ef8e.png)


This tutorial demonstrates step-by-step how to perform model quantization using the OpenVINO [Post-Training Optimization Tool (POT)](https://docs.openvino.ai/latest/pot_introduction.html), compare model accuracy between the FP32 precision and quantized INT8 precision models and run a demo of model inference based on sample code from [Ultralytics Yolov5](https://github.com/ultralytics/yolov5) with the OpenVINO backend.

## Notebook Contents

The notebook uses [Ultralytics Yolov5](https://github.com/ultralytics/yolov5) to obtain the YOLOv5-m model in OpenVINO Intermediate Representation (IR) format. Then, the OpenVINO [Post-Training Optimization Tool (POT)](https://docs.openvino.ai/latest/pot_introduction.html) API is used to quantize the model based on Non-Max Suppression (NMS) processing provided by Ultralytics. To ensure minimal accuracy loss, the accuracy is compared between the FP32 model and the INT8 model quantized by POT using "DefaultQuantization" algorithm. Finally, the code sample [detect.py](https://github.com/ultralytics/yolov5/blob/master/detect.py) from Ultralytics is used to perform inference the INT8 model and check performance using OpenVINO with sync API enabled.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
