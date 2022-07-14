# Object Detection Quantization

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F111-detection-quantization%2F111-detection-quantization.ipynb)

This tutorial shows how to quantize an object detection model, using
[Post-Training Optimization Tool API](https://docs.openvino.ai/2021.4/pot_compression_api_README.html) in OpenVINO.
For demonstration purposes, a very small dataset of 10 images presenting people at the airport is used. The images have been resized from the original resolution of 1920x1080 to 960x540. For any real use cases, a representative dataset of about 300 images would have to be applied.
The tutorial uses the [person-detection-retail-0013](
https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-detection-retail-0013) model.

## Notebook Contents

The tutorial consists of the following steps:

* Quantizing the model with POT.
* Comparing the mAP metric on `FP32` and `INT8` models.
* Visually comparing results on `FP32` and `INT8` models with annotated boxes.
* Measuring and comparing the performance of the models.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

