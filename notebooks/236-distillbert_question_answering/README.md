# Extractive Question Answering with DistilBERT and OpenVino
[DistillBERT](https://paperswithcode.com/method/distillbert) is a smaller and faster version of BERT, a popular Transformer model for natural language processing. It is trained using knowledge distillation, a technique to compress a large model (the teacher) into a smaller model (the student). DistilBERT can be used for various NLP tasks such as question answering, text classification, sentiment analysis and more.
In this notebook we will run the DistillBERT model with OpenVINO to answer questions given a context

## Notebook Contents

This notebook demonstrates how to perform extractive question answering with using the open-source DistillBERT model from HuggingFace. 

You can find more information about this model in the [research paper](https://paperswithcode.com/method/distillbert), the model is downloaded on HuggingFace [repository](https://huggingface.co/distilbert-base-cased-distilled-squad).

In this notebook we will use its capabilities for answering questions.
Notebook contains following steps:
1. Download the model
2. Perform Inference On the Model
2. Convert Model to ONNX
3. Export ONNX model and convert to OpenVINO IR using the Model Optimizer tool
4. Run the DistillBERT question answering pipeline with OpenVINO



The image below shows an example of the DistillBERT model answering questions within a given context.

![question_answering](https://user-images.githubusercontent.com/60800164/225013797-f1f24f74-8d7e-4279-bf07-44f1ea5e2aa3.jpg)

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
