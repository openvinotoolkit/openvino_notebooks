# Live Pedestrian Tracking with OpenVINO™

![Pedestrian Tracking](https://user-images.githubusercontent.com/91237924/210479548-b70dbbaa-5948-4e49-b48e-6cb6613226da.gif)

This notebook shows a pedestrian tracking scenario: it reads frames from an input video sequence, detects pedestrians in the frames, uniquely identify each one of them and track all of them until they leave the frame.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F407-pedestrian-tracking-webcam%2F407-pedestrian-tracking.ipynb)

*Binder is a free service where the webcam will not work, and performance on the video will not be good. For the best performance, install the notebooks locally.*

## Notebook Contents

In this case, We use the [Deep SORT](https://arxiv.org/abs/1703.07402) algorithm to perform object tracking.
[person detection model](https://docs.openvino.ai/nightly/omz_models_model_person_detection_0202.html) is deployed to detect the person in each frame of the video, and [reidentification model](https://docs.openvino.ai/nightly/omz_models_model_person_reidentification_retail_0287.html) is used to ouput embedding vector to match a pair of person images by the cosine distance.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

### See Also

* [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
* [Model Optimizer](https://docs.openvino.ai/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Image Processing Demo](https://docs.openvino.ai/latest/omz_demos_image_processing_demo_cpp.html)