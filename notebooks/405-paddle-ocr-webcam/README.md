# PaddleOCR with OpenVINO™

![paddleocr-webcam](https://raw.githubusercontent.com/yoyowz/classification/master/images/paddleocr.gif)

PaddleOCR performs the Optical Character Recognition (OCR) function from a video, an image, or a scanned document. It is mainly composed of three parts: DB text detection, detection frame correction and CRNN text recognition. For more details, refer to the PaddleOCR technical [article](https://arxiv.org/abs/2009.09941).

## Notebook Contents

This notebook demonstrates live paddleOCR inference with OpenVINO, using the ["Chinese and English ultra-lightweight PP-OCR model (9.4M)"](https://github.com/PaddlePaddle/PaddleOCR) from [PaddleOCR Github](https://github.com/PaddlePaddle/PaddleOCR) or [PaddleOCR Gitee](https://gitee.com/paddlepaddle/PaddleOCR). Both text detection and recognition results are visualized in a window, and text recognition results include both recognized text and its corresponding confidence level. The notebook shows how to create the following pipeline:

<p align="center" width="100%">
    <img width="80%" src="https://raw.githubusercontent.com/yoyowz/classification/master/images/pipeline.png">
</p>

Final part of this notebook shows live inference results from a webcam. Additionally, you can also upload a video file.

> **NOTE**: To use a webcam, you must run this Jupyter notebook on a computer with a webcam. If you run on a server, the webcam will not work. However, you can still do inference on a video file in the final step.

For more information about the other PaddleOCR pre-trained models, refer to the [PaddleOCR Github](https://github.com/PaddlePaddle/PaddleOCR) or [PaddleOCR Gitee](https://gitee.com/paddlepaddle/PaddleOCR).

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
