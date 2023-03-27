# Quantize Speech Recognition Models with OpenVINO™ Post-Training Optimization Tool 

This tutorial demonstrates how to apply `INT8` quantization to the speech recognition models,
using the [Post-Training Optimization Tool API (POT API)](https://docs.openvino.ai/latest/pot_compression_api_README.html)
(part of [OpenVINO Toolkit](https://docs.openvino.ai/)).

Supported models:
* `107-speech-recognition-wav2vec2.ipynb` demonstrates how to apply post-training `INT8` quantization on a fine-tuned [Wav2Vec2-Base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) [PyTorch](https://pytorch.org/) model, trained on the [LibriSpeech ASR corpus](https://www.openslr.org/12).
* `107-speech-recognition-data2vec.ipynb` demonstrates how to apply post-training `INT8` quantization on a fine-tuned [Data2Vec-Audio-Base-960h](https://huggingface.co/facebook/data2vec-audio-base-960h) [PyTorch](https://pytorch.org/) model, trained on the [LibriSpeech ASR corpus](https://www.openslr.org/12).

The code of the tutorials is designed to be extendable to custom models and datasets.

## Notebook Contents

The tutorial consists of the following steps:

* Downloading and preparing the model and dataset.
* Defining data loading and accuracy validation functionality.
* Preparing the model for quantization.
* Running optimization pipeline.
* Comparing performance of the original and quantized models.
* Compare accuracy of the original and quantized models.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
