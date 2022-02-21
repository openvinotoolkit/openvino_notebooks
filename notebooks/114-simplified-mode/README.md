# Accelerate Inference of ResNet20 Image Classification Model using the Simplified Mode of OpenVINO Post-Training Optimization Tool 

This tutorial demostrates how to apply INT8 quantization to the
Image classification model ResNet20, using the [Post-Training Optimization
Tool
Simplified Mode](https://github.com/openvinotoolkit/openvino/blob/071ce6755b02bf249dd6680f9f327bade459aeb9/tools/pot/docs/SimplifiedMode.md)
(part of [OpenVINO](https://docs.openvinotoolkit.org/)). We will use [ResNet20](https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/resnet.py) and Cifar10 dataset.
The code of the tutorial is designed to be extendable to custom models and
datasets. It consists of the following steps:

- Download and prepare the ResNet20 model and calibration dataset
- Prepare the model for quantization
- Compress the model using the simplified mode
- Compare performance of the original and quantized models
- Demonstrate the results of the optimized model
