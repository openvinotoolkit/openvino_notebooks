# Handwritten OCR with OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fhandwritten-ocr%2Fhandwritten-ocr.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/handwritten-ocr/handwritten-ocr.ipynb)

<img width="510" alt="handwritten_simplified_chinese_test" src="https://user-images.githubusercontent.com/36741649/132660640-da2211ec-c389-450e-8980-32a75ed14abb.png">

的人不一了是他有为在责新中任自之我们

This tutorial demonstrates optical character recognition for handwritten Chinese (simplified) and Japanese. An OCR tutorial for the Latin alphabet is available in the [optical character recognition notebook](../optical-character-recognition)

## Notebook Contents

This notebook provides a tutorial on how to use OCR for handwritten Japanese and simplified Chinese. Models used for this notebook are [`handwritten-japanese-recognition-0001`](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/handwritten-japanese-recognition-0001/README.md) and [`handwritten-simplified-chinese-0001`](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/handwritten-simplified-chinese-recognition-0001/README.md). To decode model output to readable text [`kondate_nakayosi`](https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/kondate_nakayosi.txt) and [`scut_ept`](https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/scut_ept.txt) charlists are used. Both models are available from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/).

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/handwritten-ocr/README.md" />
