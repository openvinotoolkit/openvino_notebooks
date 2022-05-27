# Quantize Speech Recognition Models with OpenVINO Post-Training Optimization Tool 

This tutorial demonstrates how to apply INT8 quantization to the speech recognition model
known as [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2),
using the [Post-Training Optimization Tool API (POT API)](https://docs.openvino.ai/latest/pot_compression_api_README.html)
(part of the [OpenVINO Toolkit](https://docs.openvino.ai/)).
We will use a fine-tuned [Wav2Vec2-Base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) [PyTorch](https://pytorch.org/)
model trained on the [LibriSpeech ASR corpus](https://www.openslr.org/12).
The tutorial is designed to be extendable to custom models and datasets.
It consists of the following steps:

- Download and prepare the Wav2Vec2 model and LibriSpeech dataset
- Define data loading and accuracy validation functionality
- Prepare the model for quantization
- Run optimization pipeline
- Compare performance of the original and quantized models


