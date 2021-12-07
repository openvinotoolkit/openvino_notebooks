# Object Detection Quantization

This tutorial shows how to quantize an object detection model with OpenVINO's
[Post-Training Optimization Tool
API](https://docs.openvino.ai/2021.4/pot_compression_api_README.html). The
model that is used is [person-detection-retail-0013](
https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-detection-retail-0013)
from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/)

The tutorial covers:

- Quantizing the model with POT
- Comparing the mAP metric on FP32 and INT8 models
- Visually comparing results on FP32 and INT8 models with annotated boxes
- Measuring and comparing the performance of the models

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.

