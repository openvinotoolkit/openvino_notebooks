# Grammatical Error Correction with OpenVINO


Grammatical Error Correction (GEC) is the task of correcting different types of errors in text such as spelling, punctuation, grammatical and word choice errors. 
GEC is typically formulated as a sentence correction task. A GEC system takes a potentially erroneous sentence as input and is expected to transform it into a more correct version. See the example given below:

| Input (Erroneous)                                         | Output (Corrected)                                       |
| --------------------------------------------------------- | -------------------------------------------------------- |
| I like to rides my bicycle. | I like to ride my bicycle. |

This tutorial shows how to perform grammatical error correction using OpenVINO. We will use pre-trained models from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library. To simplify the user experience, the [Hugging Face Optimum](https://huggingface.co/docs/optimum) library is used to convert the models to OpenVINO™ IR format.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert models from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino).
- Create an inference pipeline for grammatical error checking
- Optimize grammar correction pipeline with [NNCF](https://github.com/openvinotoolkit/nncf/) quantization.
- Compare original and optimized pipelines from performance and accuracy standpoints.

As the result, will be created inference pipeline which accepts text with grammatical errors and provides text with corrections as output.

The result of work represented in the table below

| Input Text                                                | Output (Corrected)                                       |
| --------------------------------------------------------- | -------------------------------------------------------- |
| Most of the course is about semantic or  content of language but there are also interesting topics to be learned from the service features except statistics in characters in documents. |  Most of the course is about the semantic content of language but there are also interesting topics to be learned from the service features except statistics in characters in documents. |

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
