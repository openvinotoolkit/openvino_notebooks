# Background removal with RMBG v1.4 and OpenVINO
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2F291-rmbg-background-removal%2F291-rmbg-background-removal.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/291-rmbg-background-removal/291-rmbg-background-removal.ipynb)


Background matting is the process of accurately estimating the foreground object in images and videos. It is a very important technique in image and video editing applications, particularly in film production for creating visual effects. In case of image segmentation, we segment the image into foreground and background by labeling the pixels. Image segmentation generates a binary image, in which a pixel either belongs to foreground or background. However, Image Matting is different from the image segmentation, wherein some pixels may belong to foreground as well as background, such pixels are called partial or mixed pixels. In order to fully separate the foreground from the background in an image, accurate estimation of the alpha values for partial or mixed pixels is necessary.

RMBG v1.4 is background removal model, designed to effectively separate foreground from background in a range of categories and image types. This model has been trained on a carefully selected dataset, which includes: general stock images, e-commerce, gaming, and advertising content, making it suitable for commercial use cases powering enterprise content creation at scale. The accuracy, efficiency, and versatility currently rival leading source-available models.

More details about model can be found in [model card](https://huggingface.co/briaai/RMBG-1.4).

![background.gif](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/a2fdaeec-b7a3-45f5-b307-ca89d447094d)

In this tutorial we consider how to convert and run this model using OpenVINO.

## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to launch RMBG model for background removal using OpenVINO. The tutorial consists of following parts:

- Download model
- Run PyTorch model inference
- Convert PyTorch model to OpenVINO IR format
- Run OpenVINO model inference
- Launch interactive demo 


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/rmbg-background-removal/README.md" />
