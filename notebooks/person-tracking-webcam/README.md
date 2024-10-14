# Live Person Tracking with OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fperson-tracking-webcam%2Fperson-tracking.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/person-tracking-webcam/person-tracking.ipynb)

*Binder is a free service where the webcam will not work, and performance on the video will not be good. For the best performance, install the notebooks locally.*

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/91237924/210479548-b70dbbaa-5948-4e49-b48e-6cb6613226da.gif">
</p>

This notebook shows a person tracking scenario: it reads frames from an input video sequence, detects people in the frames, uniquely identifies each one of them and tracks all of them until they leave the frame.

## Notebook Contents

This tutorial uses the [Deep SORT](https://arxiv.org/abs/1703.07402) algorithm to perform object tracking.
[person detection model]( https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-detection-0202/README.md) is deployed to detect the person in each frame of the video, and [reidentification model]( https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0287/README.md) is used to output embedding vector to match a pair of images of a person by the cosine distance.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

### See Also

* [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
* [Model Conversion API](https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-to-ir.html)
* [Pedestrian Tracker C++ Demo](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/pedestrian_tracker_demo/cpp/README.md)

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/person-tracking-webcam/README.md" />
