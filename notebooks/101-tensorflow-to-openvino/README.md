# TensorFlow to OpenVINO™ Model Conversion Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-to-openvino%2F101-tensorflow-to-openvino.ipynb)

<img src="https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg" width=300>

This tutorial explains how to convert [TensorFlow](www.tensorflow.org) models to OpenVINO IR. The notebook shows how to convert the [TensorFlow MobilenetV3 model](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) and then classify an image with OpenVINO Runtime.

## Notebook Contents

The notebook uses [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert the same [MobilenetV3](https://docs.openvino.ai/latest/omz_models_model_mobilenet_v3_small_1_0_224_tf.html) model used in the [001-hello-world notebook](../001-hello-world/001-hello-world.ipynb).

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
