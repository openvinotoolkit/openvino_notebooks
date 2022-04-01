# Deblurring of Image with DeblurGAN-v2 and OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jainsid2305/openvino_notebooks/bf0c665949af335aabaa21ea7b1b5274b218f0c4?urlpath=lab%2Ftree%2Fnotebooks%2F221-deblurgan-v2%2Fdeblurgan-v2.ipynb)

An image deblurring is a recovering process that recovers a sharp latent image from a blurred image, which is caused by camera shake or object motion. 

## Contents of this Notebook

### Model

This tutorial shows the implementation of Single Image Motion Deblurring with DeblurGAN-v2 in OpenVINO. This is done by first converting the DeblurGANv2 model to OpenVINO's Intermediate Representation (IR) format. I have used the [Deblurgan-v2 Model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deblurgan-v2) from [Public Pre-Trained Models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md) in [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/).

### Data
The image input taken above is a blurred image which after processing gives the deblurred image.


## Installation Instructions

If you have not done so already, please follow the [Installation Guide](/../../README.md) to install all required dependencies.