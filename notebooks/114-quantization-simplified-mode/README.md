# Simplified Post-Training Quantization of Image Classification Models with OpenVINO™ 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F114-quantization-simplified-mode%2F114-quantization-simplified-mode.ipynb)

This tutorial demonstrates how to perform `INT8` quantization with an image classification model, using the [Simplified Mode in Post-Training Optimization
Tool ](https://docs.openvino.ai/2023.0/pot_docs_simplified_mode.html) (part of [OpenVINO](https://docs.openvino.ai/)). A [ResNet20](https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/resnet.py) model and [Cifar10](http://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) dataset are used. The code in this tutorial is designed to be extendable to custom models and datasets. 

## Notebook Contents

The tutorial consists of the following steps:

* Downloading and preparing the ResNet20 model and the calibration dataset.
* Preparing the model for quantization.
* Compressing the model by using the simplified mode.
* Comparing performance of the original and quantized models.
* Demonstrating the results of the optimized model.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).