# Image Colorization with OpenVINO Tutorial

![Let there be color](https://user-images.githubusercontent.com/18904157/180923280-9caefaf1-742b-4d2f-8943-5d4a6126e2fc.png)

The idea of colorization is given a grayscale image as input, the model hallucinates a plausible, vibrant & realistic colorized version of the image.
"#### About Colorization-v2\n",
    "* The colorization-v2 model is one of the colorization group of models designed to perform image colorization.\n",
    "* Model was trained on ImageNet dataset.\n",
    "* Model consumes as input L-channel of LAB-image and give as output predict A- and B-channels of LAB-image.\n",
    "\n",
    "#### About Colorization-siggraph\n",
    "* The colorization-siggraph model is one of the colorization group of models designed to real-time user-guided image colorization.\n",
    "* Model was trained on ImageNet dataset with synthetically generated user interaction.\n",
    "* Model consumes as input L-channel of LAB-image and yields output predict A- and B-channels of LAB-image.\n",
    "\n",
    "Check out [colorization](https://github.com/richzhang/colorization) repository for more details."
## Notebook Contents

This notebook demonstrates how to colorize images with OpenVINO using the Colorization model [colorization-v2](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/colorization-v2/README.md) or [colorization-siggraph](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/colorization-siggraph) from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md) based on the paper [Colorful Image Colorization](https://arxiv.org/abs/1603.08511).

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/215-image-inpainting/README.md) to install all required dependencies.