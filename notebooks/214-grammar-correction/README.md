# Grammatical Error Correction with OpenVINO


Grammatical Error Correction (GEC) is the task of correcting different types of errors in text such as spelling, punctuation, grammatical and word choice errors. 
GEC is typically formulated as a sentence correction task. A GEC system takes a potentially erroneous sentence as input and is expected to transform it into a more correct version. See the example given below:

| Input (Erroneous)                                         | Output (Corrected)                                       |
| --------------------------------------------------------- | -------------------------------------------------------- |
| I like to rides my bicycle. | I like to ride my bicycle. |

## Notebook Contents

This tutorial shows how to perform grammatical error correction using OpenVINO and then quantize grammar correction model with [NNCF](https://github.com/openvinotoolkit/nncf) to improve its performance. We will use pre-trained models from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library. To simplify the user experience, the [Hugging Face Optimum](https://huggingface.co/docs/optimum) library is used to convert the models to OpenVINO™ IR format.

This folder contains two notebooks that show how to convert and quantize model with OpenVINO:

1. [Convert Grammar Correction model using OpenVINO](214-grammar-correction-convert.ipynb)
2. [Quantize OpenVINO Grammar Correction model using NNCF](214-grammar-correction-quantize.ipynb)

### Convert model using OpenVINO

The first notebook consists of the following steps:

- Install prerequisites
- Download and convert models from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino).
- Create an inference pipeline for grammatical error checking

As the result, will be created inference pipeline which accepts text with grammatical errors and provides text with corrections as output.

The result of work represented in the table below

| Input Text                                                                                                                                                                                                                                                                                                               | Output (Corrected)                                       |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| -------------------------------------------------------- |
| Most of the course is about semantic or  content of language but there are also interesting topics to be learned from the servicefeatures except statistics in characters in documents.At this point, He introduces herself as his native English speaker and goes on to say that if you contine to work on social scnce |  Most of the course is about the semantic content of language but there are also interesting topics to be learned from the service features except statistics in characters in documents. At this point, she introduces herself as a native English speaker and goes on to say that if you continue to work on social science, you will continue to be successful. |

### Quantize OpenVINO model using NNCF
The goal of the second notebook is to demonstrate how to speed up the model by applying 8-bit post-training quantization from [NNCF](https://github.com/openvinotoolkit/nncf/) (Neural Network Compression Framework) and infer quantized model via OpenVINO™ Toolkit. The optimization process contains the following steps:

1. Quantize the converted OpenVINO model from [214-grammar-correction-convert notebook](214-grammar-correction-convert.ipynb) with NNCF.
2. Check model result for the sample text.
3. Compare model size, performance and accuracy of FP32 and quantized INT8 models.

| Input Text                                                                                                                                                                                                                                                                                                               | Output (Quantized)                                                                                                                                                                       |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Most of the course is about semantic or  content of language but there are also interesting topics to be learned from the servicefeatures except statistics in characters in documents.At this point, He introduces herself as his native English speaker and goes on to say that if you contine to work on social scnce | Most of the course is about the semantic content of language but there are also interesting topics to be learned from the service features except statistics in characters in documents. At this point, she introduces himself as a native English speaker and goes on to say that if you continue to work on social science, you will continue to do so. |

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
