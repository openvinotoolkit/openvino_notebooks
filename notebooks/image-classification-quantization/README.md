# Accelerate Inference of MobileNet V2 Image Classification Model with NNCF in OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fimage-classification-quantization%2Fimage-classification-quantization.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/image-classification-quantization/image-classification-quantization.ipynb)

This tutorial demonstrates how to apply `INT8` quantization to the MobileNet V2 Image Classification model, using the
[NNCF Post-Training Quantization API](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training.html). The tutorial uses [MobileNetV2](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html) and [Cifar10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
The code of the tutorial is designed to be extendable to custom models and datasets.

## Notebook Contents

The tutorial consists of the following steps:

- Prepare the model for quantization.
- Define a data loading functionality.
- Perform quantization.
- Compare accuracy of the original and quantized models.
- Compare performance of the original and quantized models.
- Compare results on one picture.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/image-classification-quantization/README.md" />
