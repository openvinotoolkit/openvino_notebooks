# Selfie Segmentation using TFLite and OpenVINO
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eaidova/openvino_notebooks/blob/ea/selfie_segm/notebooks/243-tflite-selfie-segmentation/243-tflite-selfie-segmentation.ipynb)

Selfie segmentation pipeline allows developers to easily separate the background from users within a scene and focus on what matters. Adding cool effects to selfies or inserting your users into interesting background environments has never been easier. Beside photo editing, this technology is also important for video conferencing. It helps to blur or replace background during video calls.

In this tutorial we consider how to implement selfie segmentation using OpenVINO. We will use [Multiclass Selfie-segmentation model](https://developers.google.com/mediapipe/solutions/vision/image_segmenter/#multiclass-model) provided as part of [Google Mediapipe](https://developers.google.com/mediapipe) solution.

Multiclass Selfie-segmentation model is a multiclass semantic segmentation model and classifies each pixel as background, hair, body, face, clothes, others (e.g. accesories). Model supports single or multiple people in the frame, selfies and full body images. The model is based on [Vision Transformer](https://arxiv.org/abs/2010.11929) with customized boleneck and decoder architecture for real-time performance. More details about model can be found in [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf). This model is represented in Tensorflow Lite format. [TensorFlow Lite](https://www.tensorflow.org/lite/guide), often referred to as TFLite, is an open source library developed for deploying machine learning models to edge devices.


## Notebook Contents

The tutorial consist of following steps:

1. Download TFLite model and convert it to OpenVINO IR format.
2. Run inference on image.
3. Run interactive background blurring demo on video.

In this demonstration, you will see how to apply model on image and posptrocess its output in different ways (get segmentation mask, replace background, blur background).

![image-result.png](https://user-images.githubusercontent.com/29454499/251086501-cb731d92-1d43-4ead-b635-997f92603761.png)

Also, you will be able to run model inference on video from file or live webcam for blurring background scenario.

![video-background.gif](https://user-images.githubusercontent.com/29454499/251085926-14045ebc-273b-4ccb-b04f-82a3f7811b87.gif)

## Installation Instructions
If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).