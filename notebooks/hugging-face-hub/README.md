# 🤗 Hugging Face Model Hub with OpenVINO™
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fhugging-face-hub%2Fhugging-face-hub.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/hugging-face-hub/hugging-face-hub.ipynb)

The Hugging Face (HF) Model Hub is a central repository for pre-trained deep learning models. It allows exploration and provides access to thousands of models for a wide range of tasks, including text classification, question answering, and image classification.
Hugging Face provides Python packages that serve as APIs and tools to easily download and fine tune state-of-the-art pretrained models, namely [transformers] and [diffusers] packages.

![](https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/logo/hf_intel_logo.png)

## Contents: 
Throughout this notebook we will learn:
1. How to load a HF pipeline using the `transformers` package and then convert it to OpenVINO.
2. How to load the same pipeline using Optimum Intel package.

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/hugging-face-hub/README.md" />
