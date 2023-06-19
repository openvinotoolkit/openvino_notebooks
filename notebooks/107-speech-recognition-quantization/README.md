# Quantize Speech Recognition Models using NNCF PTQ API 

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

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
