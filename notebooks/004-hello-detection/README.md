# Introduction to Detection in OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F004-hello-detection%2F004-hello-detection.ipynb)

|                                                                                                                             |                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://user-images.githubusercontent.com/36741649/128489910-316aec49-4892-46f1-9e3c-b9d3646ef278.jpg" width=300> | <img src="https://user-images.githubusercontent.com/36741649/128489933-bf215a3f-06fa-4918-8833-cb0bf9fb1cc7.jpg" width=300> |


## Notebook Contents

In this basic introduction to detection with OpenVINO, the [horizontal-text-detection-0001](https://docs.openvino.ai/latest/omz_models_model_horizontal_text_detection_0001.html) model from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/) is used. It detects text in images and returns blob of data in shape of [100, 5]. For each detection, a description is in the [x_min, y_min, x_max, y_max, conf] format.

The first notebook 004-hello-detection.ipynb demonstrates how to do inference with above mentioned detection model on a single image.

The second notebook, 004-hello-detection-fiftyone.ipynb, uses the same model for the horizontal text detection on the [Total Text Dataset](https://www.kaggle.com/datasets/ipythonx/totaltextstr) for inference with the help of the open source tool FiftyOne. The notebook goes through the steps of loading the dataset of images with groundtruth detections into [FiftyOne](https://docs.voxel51.com/index.html) App, visualizing the images and adding predictions from the model and evaluating those predictions against the ground truth. 

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
