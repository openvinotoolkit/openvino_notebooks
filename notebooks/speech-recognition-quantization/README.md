# Quantize Speech Recognition Models using NNCF PTQ API
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/speech-recognition-quantization/speech-recognition-quantization-wav2vec2.ipynb)

This tutorial demonstrates how to apply `INT8` quantization to the speech recognition models,
using post-training quantization with [NNCF](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training.html) (Neural Network Compression Framework).

The code of the tutorial is designed to be extendable to custom models and datasets.

## Notebook Contents

The tutorial consists of the following steps:

* Downloading and preparing the model and dataset.
* Defining data loading and accuracy validation functionality.
* Preparing the model for quantization.
* Running quantization.
* Comparing performance of the original and quantized models.
* Compare accuracy of the original and quantized models.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/speech-recognition-quantization/README.md" />
