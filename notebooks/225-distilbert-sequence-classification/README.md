# Sequence Classification with OpenVINO API 2.0
Sentiment analysis or Sequence Classification is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. This notebook performs sequence classification using OpenVINO API 2.0.  We'll be using distilbert-base-uncased-finetuned-sst-2-english transformers based model from Huggingface. Details of the model can be found [here](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english). You can alternatively replace the model with the other bert-based models for sequence classification. The model will predict one from two classes: Positive or Negative for classifying any given text. The notebook will also provide the entire time required for the entire inference to happen. 

![image](https://user-images.githubusercontent.com/95271966/206130638-d9847414-357a-4c79-9ca7-76f4ae5a6d7f.png)

## Notebook Contents
This notebook performs sequence classification using OpenVINO API 2.0 with [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) transformers based model from Huggingface. 

## Installation Instructions
If you have not installed all required dependencies, please follow the [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md).
