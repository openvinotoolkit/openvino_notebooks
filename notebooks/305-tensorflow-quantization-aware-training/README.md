# Optimizing TensorFlow models with Neural Network Compression Framework of OpenVINO by 8-bit quantization.

This tutorial demonstrates how to use [NNCF](https://github.com/openvinotoolkit/nncf) 8-bit quantization to optimize the 
[TensorFlow](https://www.tensorflow.org) model for inference with [OpenVINO Toolkit](https://docs.openvinotoolkit.org/). 
For more advanced usage refer to these [examples](https://github.com/openvinotoolkit/nncf/tree/develop/examples).

To make downloading and training fast, we use a [ResNet-18](https://arxiv.org/abs/1512.03385) model with the 
[Imagenette](http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf) dataset. Imagenette is a subset of 10 easily classified classes from the ImageNet dataset.

This tutorial consists of the following steps:
- Fine-tuning of FP32 model
- Transform the original FP32 model to INT8
- Use fine-tuning to restore the accuracy
- Export optimized and original models to Frozen Graph and then to OpenVINO
- Measure and compare the performance of the models

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
