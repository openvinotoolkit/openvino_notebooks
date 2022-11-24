# Live Style Transfer with OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F404-style-transfer-webcam%2F404-style-transfer.ipynb)

*Binder is a free service where the webcam will not work, and performance on the video will not be good. For the best performance, install the notebooks locally.*

![style transfer](https://user-images.githubusercontent.com/109281183/203772234-f17a0875-b068-43ef-9e77-403462fde1f5.gif)

Artistic style transfer mixes the content of an image with the style of another image.

## Notebook Contents

This notebook demonstrates style transfer with OpenVINO, using the [fast-neural-style-mosaic-onnx model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/fast-neural-style-mosaic-onnx) model from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo).
The idea is to use feed-forward convolutional neural networks to generate image transformations. The networks are trained using perceptual loss functions and effectively apply style transfer on a image / video.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

### See Also

* [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
* [Model Optimizer](https://docs.openvino.ai/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Image Processing Demo](https://docs.openvino.ai/latest/omz_demos_image_processing_demo_cpp.html)