# Industrial Meter Reader

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F203-meter-reader%2F203-meter-reader.ipynb)

<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <img align='center' src= "https://user-images.githubusercontent.com/91237924/166135627-194405b0-6c25-4fd8-9ad1-83fb3a00a081.jpg" alt="drawing" height="400"/>
            </td>
            <td>
                <img align='center' src= "https://user-images.githubusercontent.com/83450930/241539765-c70db7bd-8e03-44d3-ba12-a05c33ddf6d1.png" alt="drawing" height="400"/>
            </td>
            <td>
                <img align='center' src= "https://user-images.githubusercontent.com/83450930/241539763-e3a540c1-b3db-4e42-9567-66ca7feb34dd.png" alt="drawing" height="400"/>
            </td>
        </tr>
    </table>
</html>


This notebook shows how to create an industrial meter reader with OpenVINO Runtime.

## Notebook Contents

As a routine task in a power grid, meter reading always brings a heavy workload for workers. To save the labour resources, power grids begin to implement the Deep Learning technology which enables computer to read the meter and report results.

There are two notebooks for meter reader:
- [One notebook](203-meter-reader.ipynb) uses pre-trained [PPYOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyolo) PaddlePaddle model and [DeepLabV3P](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/deeplabv3p) to build up a multiple inference task pipeline. This system will automatically detect the meters and find out their readings.
- [Digital Meter Reader](203-meter-reader-digital.ipynb) uses pre-trained [PP-OCR](https://github.com/PaddlePaddle/PaddleOCR) PaddlePaddle model to recognize the required text and reading on industrial digital meters. This system will recognize texts in specific areas and output the structured information.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
