# Accelerate Inference of NLP models with Post-Training Quantization API of NNCF

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/language-quantize-bert/language-quantize-bert.ipynb)

This tutorial demonstrates how to apply INT8 quantization to the Natural Language Processing model BERT,
using the [Post-Training Quantization API](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html).
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

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/language-quantize-bert/README.md" />
