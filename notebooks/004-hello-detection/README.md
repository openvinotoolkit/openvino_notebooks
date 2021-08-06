# Introduction to Detection in OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F004-hello-detection%2F004-hello-detection.ipynb)

|                                                                                                                             |                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://user-images.githubusercontent.com/36741649/128489910-316aec49-4892-46f1-9e3c-b9d3646ef278.jpg" width=300> | <img src="https://user-images.githubusercontent.com/36741649/128489933-bf215a3f-06fa-4918-8833-cb0bf9fb1cc7.jpg" width=300> |

This notebook demonstrates how to do inference with detection model.

## Notebook Contents

A very basic introduction to detection with OpenVINO. We use the [horizontal-text-detection-0001](https://docs.openvinotoolkit.org/latest/omz_models_model_horizontal_text_detection_0001.html) model from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/). It detects texts in images and returns blob of data in shape of [100, 5]. For each detection description has format [x_min, y_min, x_max, y_max, conf].

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
