# Live Human Pose Estimation with OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fpose-estimation-webcam%2Fpose-estimation.ipynb)

*Binder is a free service where the webcam will not work, and performance on the video will not be good. For the best performance, install the notebooks locally.*

<p align="center" width="100%">
    <img width="70%" src="https://user-images.githubusercontent.com/4547501/138267961-41d754e7-59db-49f6-b700-63c3a636fad7.gif">
</p>

Pose estimation predicts the 2D position and orientation of each person in an image or a video. Skeletons consisting of 18 predefined key points (joints) and 19 connections between them (limbs) are visualized as an overlay on the images or video.

## Notebook Contents

This notebook demonstrates human pose estimation with OpenVINO, using the OpenPose [human-pose-estimation-0001](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001) model from Open Model Zoo.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
