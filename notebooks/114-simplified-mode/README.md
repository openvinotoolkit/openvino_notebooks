# Accelerate Inference of VGG19 Image Classification Model using the Simplified Mode of OpenVINO Post-Training Optimization Tool 

This tutorial demostrates how to apply INT8 quantization to the
Image classification model VGG19, using the [Post-Training Optimization
Tool
API](https://docs.openvinotoolkit.org/latest/pot_compression_api_README.html)
(part of [OpenVINO](https://docs.openvinotoolkit.org/)). We will use [VGG19](https://arxiv.org/abs/1409.1556) and Cifar10 dataset.
The code of the tutorial is designed to be extendable to custom models and
datasets. It consists of the following steps:

- Download and prepare the VGG19 model and calibration dataset
- Prepare the model for quantization
- Compress the model using the simplified mode
- Compare performance of the original and quantized models
- Demonstrate the results of the optimized model
