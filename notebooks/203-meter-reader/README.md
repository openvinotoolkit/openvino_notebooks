# Industrial Meter Reader

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F203-meter-reader%2F203-meter-reader.ipynb)

![meter](https://user-images.githubusercontent.com/83450930/241712412-f0ab83b2-27a8-43c7-ac76-8432b9a1bac2.png)

This notebook shows how to create an industrial meter reader with OpenVINO Runtime.

## Notebook Contents

As a routine task in a power grid, meter reading always brings a heavy workload for workers. To save the labour resources, power grids begin to implement the Deep Learning technology which enables computer to read the meter and report results.

There are two notebooks for meter reader:
- [Analog Meter Reader](203-meter-reader.ipynb) uses pre-trained [PPYOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyolo) PaddlePaddle model and [DeepLabV3P](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/deeplabv3p) to build up a multiple inference task pipeline. This system will automatically detect the meters and find out their readings.
- [Digital Meter Reader](203-meter-reader-digital.ipynb) uses pre-trained [PP-OCR](https://github.com/PaddlePaddle/PaddleOCR) PaddlePaddle model to recognize the required text and reading on industrial digital meters. This system will recognize texts in specific areas and output the structured information.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
