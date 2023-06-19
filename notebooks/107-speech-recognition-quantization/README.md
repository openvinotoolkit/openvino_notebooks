# Quantize Speech Recognition Models using NNCF PTQ API
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/107-speech-recognition-quantization/107-speech-recognition-quantization-data2vec.ipynb)

This tutorial demonstrates how to apply `INT8` quantization to the speech recognition models,
using post-training quantization with [NNCF](https://docs.openvino.ai/2022.3/nncf_ptq_introduction.html) (Neural Network Compression Framework).

Supported models:
* `107-speech-recognition-wav2vec2.ipynb` demonstrates how to apply post-training `INT8` quantization on a fine-tuned [Wav2Vec2-Base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) [PyTorch](https://pytorch.org/) model, trained on the [LibriSpeech ASR corpus](https://www.openslr.org/12).
* `107-speech-recognition-data2vec.ipynb` demonstrates how to apply post-training `INT8` quantization on a fine-tuned [Data2Vec-Audio-Base-960h](https://huggingface.co/facebook/data2vec-audio-base-960h) [PyTorch](https://pytorch.org/) model, trained on the [LibriSpeech ASR corpus](https://www.openslr.org/12).

The code of the tutorials is designed to be extendable to custom models and datasets.

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
