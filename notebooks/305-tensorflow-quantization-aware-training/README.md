# Optimizing TensorFlow models with Neural Network Compression Framework of OpenVINO™ by 8-bit quantization.
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/305-tensorflow-quantization-aware-training/305-tensorflow-quantization-aware-training.ipynb)

This tutorial demonstrates how to use [NNCF](https://github.com/openvinotoolkit/nncf) 8-bit quantization to optimize the 
[TensorFlow](https://www.tensorflow.org) model for inference with [OpenVINO Toolkit](https://docs.openvino.ai/). 
For more advanced usage, refer to these [examples](https://github.com/openvinotoolkit/nncf/tree/develop/examples).

To speed up download and training, use a [ResNet-18](https://arxiv.org/abs/1512.03385) model with the 
[Imagenette](http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf) dataset. Imagenette is a subset of 10 easily classified classes from the ImageNet dataset.

## Notebook Contents

This tutorial consists of the following steps:
* Fine-tuning of `FP32` model
* Transforming the original `FP32` model to `INT8`
* Using fine-tuning to restore the accuracy.
* Exporting optimized and original models to Frozen Graph and then to OpenVINO
* Measuring and comparing the performance of the models.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
