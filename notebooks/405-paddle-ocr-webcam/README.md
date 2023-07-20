# PaddleOCR with OpenVINO™
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F405-paddle-ocr-webcam%2F405-paddle-ocr-webcam.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/405-paddle-ocr-webcam/405-paddle-ocr-webcam.ipynb)

<p align="center" width="100%">
    <img width="80%" src="https://raw.githubusercontent.com/yoyowz/classification/master/images/paddleocr.gif">
</p>

PaddleOCR performs the Optical Character Recognition (OCR) function from a video, an image, or a scanned document. It is mainly composed of three parts: DB text detection, detection frame correction and CRNN text recognition. For more details, refer to the PaddleOCR technical [article](https://arxiv.org/abs/2009.09941).

## Notebook Contents

This notebook demonstrates live PaddleOCR inference with OpenVINO, using the ["Chinese and English ultra-lightweight PP-OCRv3 model（16.2M）"](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/ppocr_introduction_en.md#pp-ocrv3) from [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) or [PaddleOCR Gitee](https://gitee.com/paddlepaddle/PaddleOCR), which includes upgraded text detector and recognizer to improve accuracy compared to its previous version, PP-OCRv2. Both text detection and recognition results are visualized in a window, and text recognition results include both recognized text and its corresponding confidence level. The notebook shows how to create the following pipeline:

<p align="center" width="100%">
    <img width="80%" src="https://raw.githubusercontent.com/yoyowz/classification/master/images/pipeline.png">
</p>


Final part of this notebook shows live inference results from a webcam. Additionally, you can also upload a video file.
> **NOTE**: To use the webcam, you must run this Jupyter notebook on a computer with a webcam. If you run on a server, the webcam will not work. However, you can still do inference on a video in the final step.

> **NOTE**: If you would like to use iGPU as your device to run the inference for PaddleOCR, note that the text recognition model within PaddleOCR is a deep learning model with dynamic input shape. Since our current release of OpenVINO 2022.2 does not support dynamic shape on iGPU, you cannot switch inference device to "GPU" for this demo. If you still want to try running inference on iGPU for PaddleOCR, it is recommended to resize the input images, for example, the bounding box images from text detection, into a fixed size to remove the dynamic input shape effect, for which some performance loss may be expected.*

For more information about the other PaddleOCR pre-trained models, refer to the [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) or [PaddleOCR Gitee](https://gitee.com/paddlepaddle/PaddleOCR).

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
