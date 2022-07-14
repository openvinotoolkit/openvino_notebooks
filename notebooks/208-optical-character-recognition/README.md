# Optical Character Recognition (OCR) with OpenVINO™

| | |
|---|---|
| <img src="https://user-images.githubusercontent.com/36741649/129315238-f1f4297e-83d0-4749-a66e-663ba4169099.jpg" width=300> | <img src="https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg" width=300> |

In this tutorial optical character recognition is presented. This notebook is a continuation of [004-hello-detection](../004-hello-detection) notebook.

## Notebook Contents

In addition to previously used [horizontal-text-detection-0001](https://docs.openvino.ai/latest/omz_models_model_horizontal_text_detection_0001.html) model, a[text-recognition-resnet](https://docs.openvino.ai/latest/omz_models_model_text_recognition_resnet_fc.html) model is used. This model reads tight aligned crop with detected text converted to a grayscale image and returns tensor that is easily decoded to predicted text. Both models are from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/).

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).