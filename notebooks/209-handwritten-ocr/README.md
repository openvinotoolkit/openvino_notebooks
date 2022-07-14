# Handwritten OCR with OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%5C209-handwritten-ocr%5C209-handwritten-ocr.ipynb)

<img width="510" alt="handwritten_simplified_chinese_test" src="https://user-images.githubusercontent.com/36741649/132660640-da2211ec-c389-450e-8980-32a75ed14abb.png">

的人不一了是他有为在责新中任自之我们

This tutorial demonstrates optical character recognition for handwritten Chinese (simplified) and Japanese. An OCR tutorial for the Latin alphabet is available in the [optical character recognition 208-notebook](../208-optical-character-recognition)

## Notebook Contents

This notebook provides a tutorial on how to use OCR for handwritten Japanese and simplified Chinese. Models used for this notebook are [handwritten-japanese-recognition](https://docs.openvino.ai/latest/omz_models_model_handwritten_japanese_recognition_0001.html) and [handwritten-simplified-chinese](https://docs.openvino.ai/latest/omz_models_model_handwritten_simplified_chinese_recognition_0001.html). To decode model output to readable text [kondate_nakayosi](https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/kondate_nakayosi.txt) and [scut_ept](https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/scut_ept.txt) charlists are used. Both models are available from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/).

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
