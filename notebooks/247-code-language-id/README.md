# Programming Language Classification with OpenVINO

**Programming language classification** is the task of identifying which programming language is used in an arbitrary code snippet. This can be useful to label new data to include in a dataset, and potentially serve as an intermediary step when input snippets need to be process based on their programming language.

More generally, this tutorial shows how to pull pre-trained models from the Hugging Face Hub using the Hugging Face [Transformers](https://huggingface.co/models) library and convert them to the OpenVINO™ IR format using the [Hugging Face Optimum](https://huggingface.co/docs/optimum) library. You will also learn how to conduct post-training quantization for OpenVINO models using Hugging Face Optimum and do some light benchmarking using the [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/index) library.

## Notebook Contents

This tutorial will be divided in 2 parts:
1. Create a simple inference pipeline with a pre-trained model using the OpenVINO™ IR format.
2. Conduct [post-training quantization](https://docs.openvino.ai/latest/ptq_introduction.html) on a pre-trained model using Hugging Face Optimum and benchmark performance.


### Licenses
[`CodeBERTa-small-v1`](https://huggingface.co/huggingface/CodeBERTa-small-v1) and [`CodeBERTa-language-id`](https://huggingface.co/huggingface/CodeBERTa-language-id): no license found on Hugging Face Hub. 

Dataset:
>Each example in the dataset is extracted from a GitHub repository, and each repository has its own license. Example-wise license information is not (yet) included in this dataset: you will need to find out yourself which license the code is using.

Additional resources:
- [Grammatical Error Correction with OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/214-grammar-correction/214-grammar-correction.ipynb)
- [Quantize a Hugging Face Question-Answering Model with OpenVINO](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/question_answering_quantization.ipynb)


## Installation Instructions

First, complete the [repository installation steps](../../README.md). Then, activate your virtual environment, or select the right Python interpreter in your IDE. 

Additional requirements will be installed directly from the notebook.