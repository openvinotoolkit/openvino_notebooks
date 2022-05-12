# PaddleOCR with OpenVINO

![paddleocr-webcam](https://raw.githubusercontent.com/yoyowz/classification/master/images/paddleocr.gif)


PaddleOCR performs the Optical Character Recognition (OCR) function from a video, an image, or a scanned document.  It is mainly composed of three parts: DB text detection, detection frame correction and CRNN text recognition. For more details, please refer to the PaddleOCR technical article (https://arxiv.org/abs/2009.09941).


## Notebook Contents

This notebook demonstrates live paddleOCR inference with OpenVINO. We use the ["Chinese and English ultra-lightweight PP-OCR model (9.4M)"](https://github.com/PaddlePaddle/PaddleOCR) from [PaddleOCR Github](https://github.com/PaddlePaddle/PaddleOCR) or [PaddleOCR Gitee](https://gitee.com/paddlepaddle/PaddleOCR). Both text detection and recognition results are visualized in a window, and text recognition results provided include both recognized text and its corresponding confidence level. In the notebook we show how to create the following pipeline:


<p align="center" width="100%">
    <img width="80%" src="https://raw.githubusercontent.com/yoyowz/classification/master/images/pipeline.png">
</p>

At the end of this notebook, you will see live inference results from your webcam. You can also upload a video file.

NOTE: _To use the webcam, you must run this Jupyter notebook on a computer with a webcam. If you run on a server, the webcam will not work. However, you can still do inference on a video in the final step._



For more information about the other PaddleOCR pre-trained models, refer to the [PaddleOCR Github](https://github.com/PaddlePaddle/PaddleOCR)  or [PaddleOCR Gitee](https://gitee.com/paddlepaddle/PaddleOCR).
 


## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
