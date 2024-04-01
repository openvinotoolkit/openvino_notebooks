# Live Style Transfer with OpenVINO™
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fstyle-transfer-webcam%2Fstyle-transfer.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/style-transfer-webcam/style-transfer.ipynb)

*Binder and Google Colab are a free services where the webcam will not work, and performance on the video will not be good. For the best performance run the notebook locally.*

![style transfer](https://user-images.githubusercontent.com/109281183/208703143-049f712d-2777-437c-8172-597ef7d53fc3.gif)

Artistic style transfer blends a single style to any given image. The real-time style transfer model is a neural network trained to apply a single style to images. Different networks can be trained for different styles you may wish to apply.

## Notebook Contents

There are five pre-trained style transfer models you can use with this notebook with the following styles: Mosaic, Rain Princess, Candy, Udnie and Pointilism. The models are downloaded from [ONNX Model Repository](https://github.com/onnx/models). They are based on the research paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) and [Instance Normalization](https://arxiv.org/abs/1607.08022). The final steps in this notebook show live inference results using video from a webcam and video file.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

## See Also

* [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
* [Model Conversion API](https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html)
* [Image Processing Demo](https://docs.openvino.ai/2024/omz_demos_image_processing_demo_cpp.html)