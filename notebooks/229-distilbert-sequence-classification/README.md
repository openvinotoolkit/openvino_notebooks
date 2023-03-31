# Sequence Classification with OpenVINO
Sequence Classification (or Text Classification) is the NLP task of predicting a label for a sequence of words.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F229-distilbert-sequence-classification%2F229-distilbert-sequence-classification.ipynb)

Sentiment analysis is a sub-task of Sequence Classification. It is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. This notebook performs sentiment analysis using OpenVINO. We will use the transformer-based [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model from Hugging Face. The Hugging Face model needs to be converted to [ONNX](https://onnx.ai/) format using the [torch.onnx.export](https://pytorch.org/docs/stable/onnx.html#example-alexnet-from-pytorch-to-onnx) function. Then, the ONNX model is converted to OpenVINO IR format. We can also replace the model with the other BERT-based models for sequence classification. The model predicts one of two classes: Positive or Negative, after analyzing the sentiment of any given text. The notebook also estimates time required for inference.

![image](https://user-images.githubusercontent.com/95271966/206130638-d9847414-357a-4c79-9ca7-76f4ae5a6d7f.png)

## Notebook Contents
This notebook performs sequence classification, using OpenVINO with the transformer-based [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model from Hugging Face. 


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
