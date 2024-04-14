# Industrial Meter Reader

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2Fmeter-reader%2Fmeter-reader.ipynb)

![meter](https://user-images.githubusercontent.com/91237924/166135627-194405b0-6c25-4fd8-9ad1-83fb3a00a081.jpg)

This notebook shows how to create an industrial meter reader with OpenVINO Runtime.

## Notebook Contents

As a routine task in a power grid, meter reading always brings a heavy workload for workers. To save the labour resources, power grids begin to implement the Deep Learning technology which enables computer to read the meter and report results.

This notebook uses pre-trained [PPYOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyolo) PaddlePaddle model and [DeepLabV3P](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/deeplabv3p) to build up a multiple inference task pipeline. This system will automatically detect the meters and find out their readings.


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
