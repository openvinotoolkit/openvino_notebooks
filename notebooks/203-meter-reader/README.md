# Industrial Meter Reader

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F203-meter-reader%2F203-meter-reader.ipynb)

![meter](https://user-images.githubusercontent.com/91237924/166135627-194405b0-6c25-4fd8-9ad1-83fb3a00a081.jpg)

This notebook shows how to create an industrial meter reader with OpenVINO Runtime.

## Notebook Contents

As a routine task in a power grid, meter reading always brings a heavy workload for workers. To save the labour resources, power grids begin to implement the Deep Learning technology which enables computer to read the meter and report results.

This notebook uses pre-trained [PPYOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyolo) PaddlePaddle model and [DeepLabV3P](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/deeplabv3p) to build up a multiple inference task pipeline. This system will automatically detect the meters and find out their readings.


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
