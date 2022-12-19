# Sequence Classification with OpenVINO
Sentiment analysis or Sequence Classification is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. This notebook performs sequence classification using OpenVINO.  We'll be using [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) transformers based model from Huggingface. The Huggingface model needs to be converted to its [ONNX](https://onnx.ai/) representation using a function [torch.onnx.export](https://pytorch.org/docs/stable/onnx.html#example-alexnet-from-pytorch-to-onnx) for the [Huggingface model](https://huggingface.co/blog/convert-transformers-to-onnx#export-with-torchonnx-low-level . The conversion to OpenVINO IR format will be done after this step.  You can alternatively replace the model with the other bert-based models for sequence classification. The model will predict one from two classes: Positive or Negative for analyzing the sentiment of any given text. The notebook will also provide the entire inference time required. 


![image](https://user-images.githubusercontent.com/95271966/206130638-d9847414-357a-4c79-9ca7-76f4ae5a6d7f.png)

## Notebook Contents
This notebook performs sequence classification using OpenVINO with [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) transformers based model from Huggingface. 


## Installation Instructions
If you have not installed all required dependencies, please follow the [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md).
