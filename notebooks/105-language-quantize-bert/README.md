# Accelerate Inference of NLP models with OpenVINO Post-Training Optimization Tool 

This tutorial demostrates how to apply INT8 quantization to the Natural
Language Processing model BERT, using the [Post-Training Optimization
Tool
API](https://docs.openvino.ai/latest/pot_compression_api_README.html)
(part of [OpenVINO](https://docs.openvino.ai/)). We will use [HuggingFace
BERT](https://huggingface.co/transformers/model_doc/bert.html)
[PyTorch](https://pytorch.org/) model fine-tuned for [Microsoft Research
Paraphrase Corpus
(MRPC)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) task.
The code of the tutorial is designed to be extendable to custom models and
datasets. It consists of the following steps:

- Download and prepare the MRPC model and dataset
- Define data loading and accuracy validation functionality
- Prepare the model for quantization
- Run optimization pipeline
- Compare performance of the original and quantized models

