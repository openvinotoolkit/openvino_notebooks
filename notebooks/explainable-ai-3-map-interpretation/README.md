# OpenVINO™ Explainable AI Toolkit: Saliency map interpretation

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/explainable-ai-3-map-interpretation/explainable-ai-3-map-interpretation.ipynb)

This is the **third notebook** in series of exploring [OpenVINO™ Explainable AI (XAI)](https://github.com/openvinotoolkit/openvino_xai/).

[OpenVINO™ Explainable AI (XAI)](https://github.com/openvinotoolkit/openvino_xai/) provides a suite of XAI algorithms for visual explanation of
[OpenVINO™](https://github.com/openvinotoolkit/openvino) Intermediate Representation (IR) models.

Using **OpenVINO XAI**, you can generate **saliency maps** that highlight regions of interest in input images from the model's perspective. This helps users understand why complex AI models produce specific responses.

This notebook shows how to use saliency maps to evaluate and debug model reasoning.

For example, it might be important to ensure that the model is using relevant features (pixels) to make a correct prediction (e.g., it might be desirable that the model is not relying on X class features to predict Y class). On the other hand, it is valuable to observe which features are used when the model is wrong.

Example: Saliency map for the `flat-coated retriever` class for the MobileNetV3 classification model. It focuses on the dog, mostly on its head and chest area, which contain the most valuable features to predict the `flat-coated retriever` class:

![Saliency Map Example](https://github.com/user-attachments/assets/5557d79d-2e9a-4784-aa17-fea2931a1435)

## Notebook Contents

The tutorial consists of the following steps:

- Prepare IR model for inference
  - Define preprocess and postprocess functions
- Explain model using ImageNet labels
- Explain for multiple images
- Saliency map examples in different usecases and their interpretations
  - True Positive High confidence
  - True Positive Low confidence
  - False Positive High confidence
  - Two mixed predictions
