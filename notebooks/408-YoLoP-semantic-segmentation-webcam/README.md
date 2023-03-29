# Live Panoptic perception using YoLoP with OpenVINO™

<p align="center" width="100%">
    <img src="https://github.com/hustvl/YOLOP/blob/d37e600cf71ecac20b08865ddfe923d76fd02d55/pictures/input1.gif">
    <img src="https://github.com/hustvl/YOLOP/blob/d37e600cf71ecac20b08865ddfe923d76fd02d55/pictures/output1.gif">
    
</p>

A panoptic driving perception system is an essential part of autonomous driving. A high-precision and real-time perception system can assist the vehicle in making the reasonable decision while driving. A panoptic driving perception network (YOLOP) to perform traffic object detection, drivable area segmentation and lane detection simultaneously has been presented. It is composed of one encoder for feature extraction and three decoders to handle the specific tasks. 

The entirety of this notebook tutorial (including the data) has been adapted from this [repository](https://github.com/hustvl/YOLOP). The authors have published their work at [Machine Intelligence Research2022](https://link.springer.com/article/10.1007/s11633-022-1339-y). 



## Notebook contents

The notebook contains a detailed tutorial of the YoLoP system starting from downloading the data from the official model repo, downloading pre - trained models and other required assets and running it against a live video stream or video source. Additional cells have been provided with functions for running it against images as well as a collection of images in a directory.


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).