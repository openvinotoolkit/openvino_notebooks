# TensorFlow to OpenVINO Model Conversion Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-to-openvino%2F101-tensorflow-to-openvino.ipynb)

![coco image](https://user-images.githubusercontent.com/15709723/127032784-8846df6f-0bfb-44ce-8920-76bcc0b5199e.jpg)

This tutorial explains how to convert [TensorFlow](www.tensorflow.org) models to OpenVINO IR with FP16 precision. Inside the notebook we show how to convert the [TensorFlow MobilenetV3 model](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) and then classify an image with OpenVINO's Inference Engine.

## Notebook Contents

The notebook uses OpenVINO [Model Optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert the same [MobilenetV3](https://docs.openvinotoolkit.org/latest/omz_models_model_mobilenet_v3_small_1_0_224_tf.html) used in the [001-hello-world notebook](../001-hello-world/001-hello-world.ipynb).

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
