# Optical Character Recognition (OCR) with OpenVINO™

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/optical-character-recognition/optical-character-recognition.ipynb)

| | |
|---|---|
| <img src="https://user-images.githubusercontent.com/36741649/129315238-f1f4297e-83d0-4749-a66e-663ba4169099.jpg" width=300> | <img src="https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg" width=300> |

In this tutorial optical character recognition is presented. This notebook is a continuation of [hello-detection](../hello-detection) notebook.

## Notebook Contents

In addition to previously used [horizontal-text-detection-0001](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/horizontal-text-detection-0001/README.md) model, [text-recognition-0014](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/text-recognition-0014/README.md) model is used. This model reads tight aligned crop with detected text converted to a grayscale image and returns tensor that is easily decoded to predicted text. Both models are from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/).

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/optical-character-recognition/README.md" />
