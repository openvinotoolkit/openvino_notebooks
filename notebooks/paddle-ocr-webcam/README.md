# PaddleOCR with OpenVINO™
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fpaddle-ocr-webcam%2Fpaddle-ocr-webcam.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/paddle-ocr-webcam/paddle-ocr-webcam.ipynb)

<p align="center" width="100%">
    <img width="80%" src="https://raw.githubusercontent.com/yoyowz/classification/master/images/paddleocr.gif">
</p>

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) is a multilingual Optical Character Recognition (OCR) toolkit built on the PaddlePaddle deep learning framework. [PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md) is its latest generation of OCR models. It consists of a DB text detection model and a multilingual CTC text recognition model with an SVTR neck, and comes in three tiers, tiny, small and medium, that trade accuracy for speed.

## Notebook Contents

This notebook demonstrates live PaddleOCR inference with OpenVINO, using the official PP-OCRv6 ONNX models from Hugging Face, for example [`PaddlePaddle/PP-OCRv6_medium_det_onnx`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) and [`PaddlePaddle/PP-OCRv6_medium_rec_onnx`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx). OpenVINO reads the ONNX models directly, without a conversion step. The pipeline of the notebook is as follows:

<p align="center" width="100%">
    <img width="80%" src="https://raw.githubusercontent.com/yoyowz/classification/master/images/pipeline.png">
</p>

The notebook contains the following steps:
1. Select the inference device and the model tier (tiny, small or medium). The tier switches both the text detection and the text recognition model.
2. Download the ONNX models and their `inference.yml` configuration files from fixed revisions of the Hugging Face repositories. The pre- and post-processing parameters and the character dictionary are read from `inference.yml`.
3. Compile the models. CPU uses dynamic input shapes. GPU, NPU and AUTO use static input shapes: frames are letterboxed into a square text detection input, and text lines are recognized with a few fixed input widths.
4. Run OCR on a frame of a test video and print the recognized text.
5. Run live OCR on the test video or a webcam. Both text detection and recognition results are visualized, and text recognition results include both the recognized text and its confidence level.

> **NOTE**: To use the webcam, you must run this Jupyter notebook on a computer with a webcam. If you run on a server, the webcam will not work. However, you can still do inference on a video in the final step.

For more information about PP-OCRv6 and other PaddleOCR models, refer to the [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) repository.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/paddle-ocr-webcam/README.md" />
