# Accelerate Inference of MobileNet V2 Image Classification Model with NNCF in OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F113-image-classification-quantization%2F113-image-classification-quantization.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/113-image-classification-quantization/113-image-classification-quantization.ipynb)

This tutorial demonstrates how to apply `INT8` quantization to the MobileNet V2 Image Classification model, using the 
[NNCF Post-Training Quantization API](https://docs.openvino.ai/2023.0/ptq_introduction.html). The tutorial uses [MobileNetV2](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html) and [Cifar10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
The code of the tutorial is designed to be extendable to custom models and datasets. 

## Notebook Contents

The tutorial consists of the following steps:

- Prepare the model for quantization.
- Define a data loading functionality.
- Perform quantization.
- Compare accuracy of the original and quantized models.
- Compare performance of the original and quantized models.
- Compare results on one picture.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
