# Live Style Transfer with OpenVINO™

![style transfer](https://user-images.githubusercontent.com/109281183/208703143-049f712d-2777-437c-8172-597ef7d53fc3.gif)

Artistic style transfer blends a single style to any given image. The real-time style transfer model is a neural network trained to apply a single style to images. Different networks can be trained for different styles you may wish to apply.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F404-style-transfer-webcam%2F404-style-transfer.ipynb)

*Binder is a free service where the webcam will not work, and performance on the video will not be good. For the best performance, install the notebooks locally.*

## Notebook Contents

There are five pre-trained style transfer models you can use with this notebook with the following styles: Mosaic, Rain Princess, Candy, Udnie and Pointilism. The models are downloaded from [ONNX Model Repository](https://github.com/onnx/models). They are based on the research paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) and [Instance Normalization](https://arxiv.org/abs/1607.08022). The final steps in this notebook show live inference results using video from a webcam and video file.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

## See Also

* [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
* [Model Optimizer](https://docs.openvino.ai/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Image Processing Demo](https://docs.openvino.ai/latest/omz_demos_image_processing_demo_cpp.html)