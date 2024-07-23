# Optical Character Recognition (OCR) with OpenVINO™

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/optical-character-recognition/optical-character-recognition.ipynb)

| | |
|---|---|
| <img src="https://user-images.githubusercontent.com/36741649/129315238-f1f4297e-83d0-4749-a66e-663ba4169099.jpg" width=300> | <img src="https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg" width=300> |

In this tutorial optical character recognition is presented. This notebook is a continuation of [hello-detection](../hello-detection) notebook.

## Notebook Contents

In addition to previously used [horizontal-text-detection-0001](https://docs.openvino.ai/2024/omz_models_model_horizontal_text_detection_0001.html) model, a[text-recognition-resnet](https://docs.openvino.ai/2024/omz_models_model_text_recognition_resnet_fc.html) model is used. This model reads tight aligned crop with detected text converted to a grayscale image and returns tensor that is easily decoded to predicted text. Both models are from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/).

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).