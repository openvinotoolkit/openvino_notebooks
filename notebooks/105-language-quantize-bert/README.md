# Accelerate Inference of NLP models with OpenVINO™ Post-Training Optimization Tool 

This tutorial demonstrates how to apply INT8 quantization to the Natural Language Processing model BERT, 
using the [Post-Training Optimization Tool API](https://docs.openvino.ai/latest/pot_compression_api_README.html)
(part of [OpenVINO](https://docs.openvino.ai/)). 
The [HuggingFace BERT](https://huggingface.co/transformers/model_doc/bert.html) [PyTorch](https://pytorch.org/) model, 
fine-tuned for [Microsoft Research Paraphrase Corpus (MRPC)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) task 
is used. The code of this tutorial is designed to be extendable to custom models and datasets. 

## Notebook Contents

The tutorial consists of the following steps:

* Downloading and preparing the MRPC model and a dataset.
* Defining data loading and accuracy validation functionality.
* Preparing the model for quantization.
* Running optimization pipeline.
* Comparing performance of the original and quantized models.
