# Accelerate Inference of MobileNet V2 Image Classification Model with Post-Training Optimization Tool in OpenVINO™

This tutorial demonstrates how to apply `INT8` quantization to the
MobileNet V2 Image Classification model, using the 
[Post-Training Optimization Tool API](https://docs.openvino.ai/latest/pot_compression_api_README.html)
(part of [OpenVINO](https://docs.openvino.ai/)). The tutorial uses [mobilenet-v2](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html) and [Cifar10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
The code of the tutorial is designed to be extendable to custom models and
datasets. 

## Notebook Contents

The tutorial consists of the following steps:

* Downloading and preparing the Mobilenet-v2 model and the dataset.
* Defining a data loading and an accuracy validation functionality.
* Preparing the model for quantization.
* Running optimization pipeline.
* Comparing performance of the original and quantized models.
