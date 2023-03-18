# SSDMobileNet v2 to OpenVINO™ Model Conversion Tutorial

This tutorial explains how to convert SSDMobileNet v2 models to OpenVINO IR. The notebook shows how to convert the [TensorFlow SSDMobileNet v2 model](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320) and then classify an image with OpenVINO Runtime.

## Notebook Contents

The notebook uses [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert the SSDMobileNet v2 available on [TensorFlow Hub](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320) to OpenVINO IR format. We then use the OpenVINO runtime api to run inference on the converted model.

## Installation Instructions
If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
