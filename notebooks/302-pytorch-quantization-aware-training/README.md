# Optimizing PyTorch models with Neural Network Compression Framework of OpenVINO™ by 8-bit quantization.

This tutorial demonstrates how to use [NNCF](https://github.com/openvinotoolkit/nncf) 8-bit quantization to optimize the 
[PyTorch](https://pytorch.org/) model for inference with [OpenVINO Toolkit](https://docs.openvino.ai/). 
For more advanced usage, refer to these [examples](https://github.com/openvinotoolkit/nncf/tree/develop/examples).

This notebook is based on 'ImageNet training in PyTorch' [example](https://github.com/pytorch/examples/blob/master/imagenet/main.py).
To speed up download and training, use a [ResNet-18](https://arxiv.org/abs/1512.03385) model with the 
[Tiny ImageNet](http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf) dataset.

## Notebook Contents

This tutorial consists of the following steps:
* Transforming the original `FP32` model to `INT8`
* Using fine-tuning to restore the accuracy.
* Exporting optimized and original models to ONNX and then to OpenVINO
* Measuring and comparing the performance of the models.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

