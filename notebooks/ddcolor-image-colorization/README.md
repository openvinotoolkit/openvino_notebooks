# Colorize grayscale images using 🎨 DDColor and OpenVINO

![](https://github.com/piddnad/DDColor/raw/master/assets/teaser.png)

Image colorization is the process of adding color to grayscale images. Initially captured in black and white, these images are transformed into vibrant, lifelike representations by estimating RGB colors. This technology enhances both aesthetic appeal and perceptual quality. Historically, artists manually applied colors to monochromatic photographs, a painstaking task that could take up to a month for a single image. However, with advancements in information technology and the rise of deep neural networks, automated image colorization has become increasingly important.

DDColor is one of the most progressive methods of image colorization in our days. It is a novel approach using dual decoders: a pixel decoder and a query-based color decoder, that stands out in its ability to produce photo-realistic colorization, particularly in complex scenes with multiple objects and diverse contexts.
![](https://github.com/piddnad/DDColor/raw/master/assets/network_arch.jpg)

More details about this approach can be found in original model [repository](https://github.com/piddnad/DDColor) and [paper](https://arxiv.org/abs/2212.11613).

In this tutorial we consider how to convert and run DDColor using OpenVINO. Additionally, we will demonstrate how to optimize this model using [NNCF](https://github.com/openvinotoolkit/nncf/).

## Notebook Contents

This notebook demonstrates Image Colorization with the [DDColor](https://github.com/piddnad/DDColor) in OpenVINO.

The tutorial consists of following steps:
- Install prerequisites
- Load and run PyTorch model inference
- Convert Model to Openvino Intermediate Representation format
- Run OpenVINO model inference on single image
- Optimize Model
- Compare results of original and optimized models
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/ddcolor-image-colorization/README.md" />
