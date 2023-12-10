# Semantic Segmentation with OpenVINO™ using Segmenter

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/204-segmenter-semantic-segmentation/204-segmenter-semantic-segmentation.ipynb)

This notebook demonstrates how to convert and use [Segmenter](https://github.com/rstrudel/segmenter) PyTorch model 
with OpenVINO.

![Segmenter example](https://user-images.githubusercontent.com/61357777/223854308-d1ac4a39-cc0c-4618-9e4f-d9d4d8b991e8.jpg)

Semantic segmentation is a difficult computer vision problem with many applications. 
Its goal is to assign labels to each pixel according to the object it belongs to, creating so-called segmentation masks.
To properly assign this label, the model needs to consider the local as well as global context of the image.
This is where transformers offer their advantage as they work well in capturing global context.
Segmenter is based on Vision Transformer working as an encoder, and Mask Transformer working as a decoder.
With this configuration, it achieves good results on different datasets such as ADE20K, Pascal Context, and Cityscapes.

![Segmenteer diagram](https://user-images.githubusercontent.com/24582831/148507554-87eb80bd-02c7-4c31-b102-c6141e231ec8.png)
> Credits for this image go to [original authors of Segmenter](https://github.com/rstrudel/segmenter).

More about the model and its details can be found in the following paper:
[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)

## Notebook Contents

The tutorial consists of the following steps:

* Preparing PyTorch Segmenter model
* Preparing preprocessing and visualization functions
* Validating inference of original model
* Converting PyTorch model to OpenVINO IR
* Validating inference of the converted model
* Benchmark performance of the converted model


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
