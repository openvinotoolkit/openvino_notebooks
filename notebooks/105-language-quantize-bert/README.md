# Accelerate Inference of NLP models with Post-Training Quantization API of NNCF 

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/105-language-quantize-bert/105-language-quantize-bert.ipynb)

This tutorial demonstrates how to apply INT8 quantization to the Natural Language Processing model BERT, 
using the [Post-Training Quantization API](https://docs.openvino.ai/nightly/basic_quantization_flow.html). 
The [HuggingFace BERT](https://huggingface.co/docs/transformers/model_doc/bert) [PyTorch](https://pytorch.org/) model, 
fine-tuned for [Microsoft Research Paraphrase Corpus (MRPC)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) task 
is used. The code of this tutorial is designed to be extendable to custom models and datasets. 

## Notebook Contents

The tutorial consists of the following steps:

* Downloading and preparing the MRPC model and a dataset.
* Defining data loading functionality.
* Running optimization pipeline.
* Comparing F1 score of the original and quantized models.
* Comparing performance of the original and quantized models.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
