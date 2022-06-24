# Quantize Speech Recognition Models with OpenVINO™ Post-Training Optimization Tool 

This tutorial demonstrates how to apply `INT8` quantization to the speech recognition model,
known as [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2),
using the [Post-Training Optimization Tool API (POT API)](https://docs.openvino.ai/latest/pot_compression_api_README.html)
(part of [OpenVINO Toolkit](https://docs.openvino.ai/)).
A fine-tuned [Wav2Vec2-Base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) [PyTorch](https://pytorch.org/)
model, trained on the [LibriSpeech ASR corpus](https://www.openslr.org/12), is used here.
The code of the tutorial is designed to be extendable to custom models and datasets.

## Notebook Contents

The tutorial consists of the following steps:

* Downloading and preparing the Wav2Vec2 model and LibriSpeech dataset.
* Defining data loading and accuracy validation functionality.
* Preparing the model for quantization.
* Running optimization pipeline.
* Comparing performance of the original and quantized models.


