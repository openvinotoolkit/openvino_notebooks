# Sequence Classification with OpenVINO
Sequence Classification (or Text Classification) is the NLP task of predicting a label for a sequence of words.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2F229-distilbert-sequence-classification%2F229-distilbert-sequence-classification.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/229-distilbert-sequence-classification/229-distilbert-sequence-classification.ipynb)

Sentiment analysis is a sub-task of Sequence Classification. It is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. This notebook performs sentiment analysis using OpenVINO. We will use the transformer-based [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model from Hugging Face converting to OpenVINO IR format using OpenVINO PyTorch Frontend and model conversion Python API. We can also replace the model with the other BERT-based models for sequence classification. The model predicts one of two classes: Positive or Negative, after analyzing the sentiment of any given text. The notebook also estimates time required for inference.

![image](https://user-images.githubusercontent.com/95271966/206130638-d9847414-357a-4c79-9ca7-76f4ae5a6d7f.png)

## Notebook Contents
This notebook performs sequence classification, using OpenVINO with the transformer-based [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model from Hugging Face. 


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
