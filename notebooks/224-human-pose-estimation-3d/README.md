# Title of Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/spencergotowork/openvino_notebooks/main)
*Binder is a free service where the webcam will not work, and performance on the video will not be good. For best performance, we recommend installing the notebooks locally.*

![pose estimation_webgl](./data/threejs.gif)
![pose estimation_opencv](./data/opencv.gif)

This demo contains 3D multi-person pose estimation demo. Intel OpenVINO™ backend can be used for fast inference on CPU. This demo is based on Lightweight OpenPose and Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB papers. It detects 2D coordinates of up to 18 types of keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees, and ankles, as well as their 3D coordinates.

## Notebook Contents

This notebook uses the model to estimate 3D human pose and draw them in 2D screen. The input source can be video files or webcam.

You can find an introduction to the model [here](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001).

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.

Make sure your [Jupyter extension](https://github.com/jupyter-widgets/pythreejs#jupyterlab) is working properly.
