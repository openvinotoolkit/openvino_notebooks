# Semantic Segmentation with OpenVINO™ using Segmenter

This notebook demonstrates how to convert and use [Segmenter](https://github.com/rstrudel/segmenter) PyTorch model 
with OpenVINO.

<div style="text-align:center">
    <img src="https://user-images.githubusercontent.com/61357777/223854308-d1ac4a39-cc0c-4618-9e4f-d9d4d8b991e8.jpg" width="70%"/>
</div>

Semantic segmentation is a difficult computer vision problem with many applications. 
Its goal is to assign labels to each pixel according to object it belongs to, creating so-called segmentation masks.
To properly assign this label, the model needs to consider local as well as global context of image.
This is where transformers offer their advantage as they work well in capturing global context.
Segmenter is based on Vision Transformer working as encoder, and Mask Transformer working as decoder.
With this configuration, it achieves good results on different datasets such as ADE20K, Pascal Context and Cityscapes.

<div style="text-align:center">
    <img src="https://user-images.githubusercontent.com/24582831/148507554-87eb80bd-02c7-4c31-b102-c6141e231ec8.png" width="70%"/>
</div>

More about the model and its details can be found in the following paper:
[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)

## Notebook Contents

The tutorial consists of the following steps:

* Preparing PyTorch Segmenter model
* Preparing preprocessing and visualization function
* Validating inference of original model
* Converting PyTorch model to ONNX
* Converting ONNX to OpenVINO IR
* Validating inference of converted model
* Validating converted model on a subset of ADE20K


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md). 
All additional required libraries will be installed inside this notebook.