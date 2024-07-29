# OpenVINO™ Explainable AI Toolkit: Saliency map interpretation

**OpenVINO™ Explainable AI (XAI) Toolkit** provides a suite of XAI algorithms for visual explanation of
[**OpenVINO™**](https://github.com/openvinotoolkit/openvino) Intermediate Representation (IR) models.

Using **OpenVINO XAI**, you can generate **saliency maps** that highlight regions of interest in input images from the model's perspective. This helps users understand why complex AI models produce specific responses.

This notebook shows an example how to use OpenVINO XAI.

It depicts a heatmap with areas of interest where neural network (classification or detection) focuses before making a decision.

Example: Saliency map for `flat-coated retriever` class for MobileNetV3 classification model:

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
