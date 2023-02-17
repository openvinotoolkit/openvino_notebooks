# Grammatical Error Correction with OpenVINO


Grammatical Error Correction (GEC) is the task of correcting different kinds of errors in text such as spelling, punctuation, grammatical, and word choice errors. 
GEC is typically formulated as a sentence correction task. A GEC system takes a potentially erroneous sentence as input and is expected to transform it into its corrected version. See the example given below: 

| Input (Erroneous)                                         | Output (Corrected)                                       |
| --------------------------------------------------------- | -------------------------------------------------------- |
|She see Tom is catched by policeman in park at last night. | She saw Tom caught by a policeman in the park last night.|

This tutorial shows how to perform grammatical error correction using OpenVINO. We will use pre-trained models from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library. To simplify the user experience, the [Hugging Face Optimum](https://huggingface.co/docs/optimum) library is used to convert the models to OpenVINO™ IR format.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert models from a public source using the OpenVINO integration with Hugging Face Optimum.
- Create an inference pipeline for grammatical error checking

As the result, will be created inference pipeline which accepts text with grammatical errors and provides text with corrections as output.

The result of work represneted in the table below

| Input Text                                                | Output (Corrected)                                       |
| --------------------------------------------------------- | -------------------------------------------------------- |
| Most of the course is about semantic or  content of language but there are also interesting topics to be learned from the servicefeatures except statistics in characters in documents. |  Most of the course is about the semantic content of language but there are also interesting topics to be learned from the service features except statistics in characters in documents. |

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
