# Post-Training Quantization and Sparsification of PyTorch models with POT

This tutorial demonstrates how to use [POT] (Post Training Optimization) 8-bit quantization in
post-training mode to optimize a [PyTorch](https://pytorch.org/) model
for high-speed inference via [OpenVINO Toolkit](https://docs.openvino.ai/).

To speed up download and validation, this tutorial uses a pre-trained [ResNet-50](https://arxiv.org/abs/1512.03385)
model on the [Tiny ImageNet](http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf) dataset.

## Notebook contents

The tutorial consists of the following steps:

* Evaluating the original model.
* Transforming the original `FP32` model to `INT8`.
* Exporting optimized and original models to ONNX and then to OpenVINO IR.
* Comparing performance of the obtained `FP32` and `INT8` models.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
