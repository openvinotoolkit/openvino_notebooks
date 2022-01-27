# Neural Style Transfer with ONNX models and OpenVINO

![Neural Style Transfer Network Output](https://user-images.githubusercontent.com/77325899/147366340-4a281b0b-066d-4114-b9ef-48634d733095.png)


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F212-onnx-style-transfer%2F212-onnx-style-transfer.ipynb)

This notebook demonstrates [Fast Neural Style Transfer](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style) on ONNX models with OpenVINO. Style Transfer models mix the content of an image with the style of another image.

For this notebook, we use five pretrained models, for the following styles: Mosaic, Rain Princess, Candy, Udnie and Pointilism. The models are from the [ONNX Model Repository](https://github.com/onnx/models) and are based on the research paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) by Justin Johnson, Alexandre Alahi and Li Fei-Fei.


## Notebook Contents

In this notebook, we:
- load an ONNX model in Inference Engine and do inference on this model,
- show inference results on five neural style transfer models,
- save the transformed images and provide a download link.

The ONNX models are downloaded in the notebook, and a sample image is provided. See [this short video](https://www.youtube.com/watch?v=1bd2QHqQSH4) for instructions on how to upload your own images to Jupyter Lab.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
