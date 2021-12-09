# Object Detection Quantization

This tutorial shows how to quantize an object detection model, using OpenVINO's
[Post-Training Optimization Tool
API](https://docs.openvino.ai/2021.4/pot_compression_api_README.html).
For demonstration purposes, we use a very small dataset of 10 images presenting people at the airport. The images have been resized from the original resolution of 1920x1080 to 960x540. For any real use cases, a representative dataset of about 300 images would have to be applied.
The model used is [person-detection-retail-0013](
https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-detection-retail-0013)

The tutorial covers:

- Quantizing the model with POT
- Comparing the mAP metric on FP32 and INT8 models
- Visually comparing results on FP32 and INT8 models with annotated boxes
- Measuring and comparing the performance of the models

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.

