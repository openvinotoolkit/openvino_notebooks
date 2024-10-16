# OpenVINO™ Explainable AI Toolkit (2/3): Deep Dive

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/explainable-ai-2-deep-dive/explainable-ai-2-deep-dive.ipynb)

This is the **second notebook** in series of exploring [OpenVINO™ Explainable AI (XAI)](https://github.com/openvinotoolkit/openvino_xai/):

1. [OpenVINO™ Explainable AI Toolkit (1/3): Basic](../explainable-ai-1-basic/README.md)
2. [**OpenVINO™ Explainable AI Toolkit (2/3): Deep Dive**](../explainable-ai-2-deep-dive/README.md)
3. [OpenVINO™ Explainable AI Toolkit (3/3): Saliency map interpretation](../explainable-ai-3-map-interpretation/README.md)

[OpenVINO™ Explainable AI (XAI)](https://github.com/openvinotoolkit/openvino_xai/) provides a suite of XAI algorithms for visual explanation of
[OpenVINO™](https://github.com/openvinotoolkit/openvino) Intermediate Representation (IR) models.

Using **OpenVINO XAI**, you can generate **saliency maps** that highlight regions of interest in input images from the model's perspective. This helps users understand why complex AI models produce specific responses.

This notebook shows an example how to use OpenVINO XAI.

It depicts a heatmap with areas of interest where neural network (classification or detection) focuses before making a decision.

Example: Saliency map for `flat-coated retriever` class for MobileNetV3 classification model:

![Saliency Map Example](https://github.com/user-attachments/assets/5557d79d-2e9a-4784-aa17-fea2931a1435)

## Notebook Contents

The tutorial consists of the following steps:

- Run `Explainer` in `AUTO` mode
- Specify preprocess and postprocess functions
- Run `Explainer` in `WHITEBOX` mode
  - Insert XAI branch to IR or PyTorch model to use updated model in own pipelines
- Run `Explainer` in `BLACKBOX` mode
- Advanced: add label names and use them to save saliency maps instead of label indexes

These are explainable AI algorithms supported by OpenVINO XAI:

| Domain          | Task                 | Type      | Algorithm           | Links |
|-----------------|----------------------|-----------|---------------------|-------|
| Computer Vision | Image Classification | White-Box | ReciproCAM          | [arxiv](https://arxiv.org/abs/2209.14074) / [src](https://github.com/openvinotoolkit/openvino_xai/blob/develop/openvino_xai/methods/white_box/recipro_cam.py) |
|                 |                      |           | VITReciproCAM       | [arxiv](https://arxiv.org/abs/2310.02588) / [src](https://github.com/openvinotoolkit/openvino_xai/blob/develop/openvino_xai/methods/white_box/recipro_cam.py) |
|                 |                      |           | ActivationMap       | experimental / [src](https://github.com/openvinotoolkit/openvino_xai/blob/develop/openvino_xai/methods/white_box/activation_map.py)                           |
|                 |                      | Black-Box | AISEClassification  | [src](https://github.com/openvinotoolkit/openvino_xai/blob/develop/openvino_xai/methods/black_box/aise/classification.py)                                     |
|                 |                      |           | RISE                | [arxiv](https://arxiv.org/abs/1806.07421v3) / [src](https://github.com/openvinotoolkit/openvino_xai/blob/develop/openvino_xai/methods/black_box/rise.py)      |
|                 | Object Detection     | White-Box | ClassProbabilityMap | experimental / [src](https://github.com/openvinotoolkit/openvino_xai/blob/develop/openvino_xai/methods/white_box/det_class_probability_map.py)                |
|                 |                      | Black-Box | AISEDetection       | [src](https://github.com/openvinotoolkit/openvino_xai/blob/develop/openvino_xai/methods/black_box/aise/detection.py)                                          |


### Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/explainable-ai-2-deep-dive/README.md" />
