# Live Person Tracking with OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F407-person-tracking-webcam%2F407-person-tracking.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/407-person-tracking-webcam/407-person-tracking.ipynb)

*Binder is a free service where the webcam will not work, and performance on the video will not be good. For the best performance, install the notebooks locally.*

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/91237924/210479548-b70dbbaa-5948-4e49-b48e-6cb6613226da.gif">
</p>

This notebook shows a person tracking scenario: it reads frames from an input video sequence, detects people in the frames, uniquely identifies each one of them and tracks all of them until they leave the frame.

## Notebook Contents

This tutorial uses the [Deep SORT](https://arxiv.org/abs/1703.07402) algorithm to perform object tracking.
[person detection model]( https://docs.openvino.ai/2023.0/omz_models_model_person_detection_0202.html) is deployed to detect the person in each frame of the video, and [reidentification model]( https://docs.openvino.ai/2023.0/omz_models_model_person_reidentification_retail_0287.html) is used to output embedding vector to match a pair of images of a person by the cosine distance.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

### See Also

* [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
* [Model Conversion API](https://docs.openvino.ai/nightly/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Pedestrian Tracker C++ Demo](https://docs.openvino.ai/2023.0/omz_demos_pedestrian_tracker_demo_cpp.html#doxid-omz-demos-pedestrian-tracker-demo-cpp)
