# Post-Training Quantization of PyTorch models with NNCF

This tutorial demonstrates how to use [NNCF](https://github.com/openvinotoolkit/nncf) 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize a [PyTorch](https://pytorch.org/) model
for high-speed inference via [OpenVINO Toolkit](https://docs.openvinotoolkit.org/). For more advanced NNCF
usage refer to these [examples](https://github.com/openvinotoolkit/nncf/tree/develop/examples).

To make downloading and validating fast, we use an already pretrained [ResNet-50](https://arxiv.org/abs/1512.03385)
model on the [Tiny ImageNet](http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf) dataset.

It consists of the following steps:

- Evaluate the original model
- Transform the original FP32 model to INT8
- Export optimized and original models to ONNX and then to OpenVINO IR
- Compare performance of the obtained FP32 and INT8 models

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
