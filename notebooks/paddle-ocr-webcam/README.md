# PaddleOCR with OpenVINO™
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fpaddle-ocr-webcam%2Fpaddle-ocr-webcam.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/paddle-ocr-webcam/paddle-ocr-webcam.ipynb)

<p align="center" width="100%">
    <img width="80%" src="https://raw.githubusercontent.com/yoyowz/classification/master/images/paddleocr.gif">
</p>

PaddleOCR performs the Optical Character Recognition (OCR) function from a video, an image, or a scanned document. It is mainly composed of three parts: DB text detection, detection frame correction and [SVTR](https://arxiv.org/abs/2205.00159) text recognition. For more details, refer to the PaddleOCR [introduction](https://github.com/PaddlePaddle/PaddleOCR/blob/4b17511491adcfd0f3e2970895d06814d1ce56cc/doc/doc_en/PP-OCRv3_introduction_en.md).

## Notebook Contents

This notebook demonstrates live PaddleOCR inference with OpenVINO, using the ["Chinese and English ultra-lightweight PP-OCRv3 model（16.2M）"](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/ppocr_introduction_en.md#pp-ocrv3) from [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) or [PaddleOCR Gitee](https://gitee.com/paddlepaddle/PaddleOCR), which includes upgraded text detector and recognizer to improve accuracy compared to its previous version, PP-OCRv2. Both text detection and recognition results are visualized in a window, and text recognition results include both recognized text and its corresponding confidence level. The notebook shows how to create the following pipeline:

<p align="center" width="100%">
    <img width="80%" src="https://raw.githubusercontent.com/yoyowz/classification/master/images/pipeline.png">
</p>


Final part of this notebook shows live inference results from a webcam. Additionally, you can also upload a video file.
> **NOTE**: To use the webcam, you must run this Jupyter notebook on a computer with a webcam. If you run on a server, the webcam will not work. However, you can still do inference on a video in the final step.


For more information about the other PaddleOCR pre-trained models, refer to the [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) or [PaddleOCR Gitee](https://gitee.com/paddlepaddle/PaddleOCR).

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/paddle-ocr-webcam/README.md" />
