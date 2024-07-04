# OpenVINO™ Explainable AI Toolkit: Classification Explanation

**OpenVINO™ Explainable AI (XAI) Toolkit** provides a suite of XAI algorithms for visual explanation of
[**OpenVINO™**](https://github.com/openvinotoolkit/openvino) Intermediate Representation (IR) models.

Given **OpenVINO** models and input images, **OpenVINO XAI** generates **saliency maps**
which highlights regions of the interest in the inputs from the models' perspective
to help users understand the reason why the complex AI models give such responses.

This notebook shows an example how to use OpenVINO XAI.

It depicts a heatmap with areas of interest where neural network (classification or detection) focuses before making a decision.

Example: Saliency map for `flat-coated retriever` class for MobileNetV3 classification model:

![Saliency Map Example](./retriever-saliency-map.jpg)

## Notebook Contents

The tutorial consists of the following steps:

- Run explainer in Auto-mode
- Specify preprocess and postprocess functions
- Run explainer in White-box mode
    - Insert XAI branch to use updated model in own pipelines
- Run explainer in Black-box mode
- Advanced: add label names and use them to save saliency maps instead of label indexes 
