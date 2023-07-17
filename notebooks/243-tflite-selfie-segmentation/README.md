# Selfie Segmentation using TFLite and OpenVINO
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F243-tflite-selfie-segmentation%2F243-tflite-selfie-segmentation.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/243-tflite-selfie-segmentation/243-tflite-selfie-segmentation.ipynb)

The Selfie segmentation pipeline allows developers to easily separate the background from users within a scene and focus on what matters. Adding cool effects to selfies or inserting your users into interesting background environments has never been easier. Besides photo editing, this technology is also important for video conferencing. It helps to blur or replace the background during video calls.

In this tutorial, we consider how to implement selfie segmentation using OpenVINO. We will use [Multiclass Selfie-segmentation model](https://developers.google.com/mediapipe/solutions/vision/image_segmenter/#multiclass-model) provided as part of [Google MediaPipe](https://developers.google.com/mediapipe) solution.

The Multiclass Selfie-segmentation model is a multiclass semantic segmentation model and classifies each pixel as background, hair, body, face, clothes, and others (e.g. accessories). The model supports single or multiple people in the frame, selfies, and full-body images. The model is based on [Vision Transformer](https://arxiv.org/abs/2010.11929) with customized bottleneck and decoder architecture for real-time performance. More details about the model can be found in the [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf). This model is represented in Tensorflow Lite format. [TensorFlow Lite](https://www.tensorflow.org/lite/guide), often referred to as TFLite, is an open-source library developed for deploying machine learning models to edge devices.


## Notebook Contents

The tutorial consists of the following steps:

1. Download the TFLite model and convert it to OpenVINO IR format.
2. Run inference on the image.
3. Run interactive background blurring demo on video.

In this demonstration, you will see how to apply the model on an image and postprocess its output in different ways (get segmentation mask, replace background, blur background).

![image-result.png](https://user-images.githubusercontent.com/29454499/251086501-cb731d92-1d43-4ead-b635-997f92603761.png)

Also, you will be able to run model inference on video from a file or live webcam for blurring background scenarios.

![video-background.gif](https://user-images.githubusercontent.com/29454499/251085926-14045ebc-273b-4ccb-b04f-82a3f7811b87.gif)

## Installation Instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).