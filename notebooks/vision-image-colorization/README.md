# Image Colorization with OpenVINO Tutorial
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fvision-image-colorization%2Fvision-image-colorization.ipynb)

![Let there be color](https://user-images.githubusercontent.com/18904157/180923280-9caefaf1-742b-4d2f-8943-5d4a6126e2fc.png)

The idea of colorization is given a grayscale image as input, the model hallucinates a plausible, vibrant & realistic colorized version of the image.

**About Colorization-v2**

* The colorization-v2 model is one of the colorization group of models designed to perform image colorization.
* Model was trained on ImageNet dataset.
* Model consumes as input L-channel of LAB-image and give as output predict A- and B-channels of LAB-image.

**About Colorization-siggraph**

* The colorization-siggraph model is one of the colorization group of models designed to real-time user-guided image colorization.
* Model was trained on ImageNet dataset with synthetically generated user interaction.
* Model consumes as input L-channel of LAB-image and yields output predict A- and B-channels of LAB-image.

Check out [colorization](https://github.com/richzhang/colorization) repository for more details.

## Notebook Contents

This notebook demonstrates how to colorize images with OpenVINO using the Colorization model [colorization-v2](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/colorization-v2/README.md) or [colorization-siggraph](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/colorization-siggraph) from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md) based on the paper: [Colorful Image Colorization](https://arxiv.org/abs/1603.08511).

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).