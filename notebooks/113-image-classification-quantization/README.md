# Accelerate Inference of Mobilenet V2 Image Classification Model with OpenVINO Post-Training Optimization Tool 

This tutorial demostrates how to apply INT8 quantization to the
Image classification model mobilenet-v2, using the [Post-Training Optimization
Tool
API](https://docs.openvino.ai/latest/pot_compression_api_README.html)
(part of [OpenVINO](https://docs.openvino.ai/)). We will use [mobilenet-v2](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html) and Cifar10 dataset.
The code of the tutorial is designed to be extendable to custom models and
datasets. It consists of the following steps:

- Download and prepare the Mobilenet-v2 model and dataset
- Define data loading and accuracy validation functionality
- Prepare the model for quantization
- Run optimization pipeline
- Compare performance of the original and quantized models
