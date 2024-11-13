# Convert models from ModelScope to OpenVINO

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/modelscope-to-openvino/modelscope-to-openvino.ipynb)

![](https://camo.githubusercontent.com/bbda58b4f77b80d9206e3410b533ca5a2582b81070e7dd283ee12fd0d442bd2b/68747470733a2f2f6d6f64656c73636f70652e6f73732d636e2d6265696a696e672e616c6979756e63732e636f6d2f6d6f64656c73636f70652e676966)

[ModelScope](https://www.modelscope.cn/home) is a “Model-as-a-Service” (MaaS) platform that seeks to bring together most advanced machine learning models from the AI community, and to streamline the process of leveraging AI models in real applications. Hundreds of models are made publicly available on ModelScope (700+ and counting), covering the latest development in areas such as NLP, CV, Audio, Multi-modality, and AI for Science, etc. Many of these models represent the SOTA in their specific fields, and made their open-sourced debut on ModelScope.

This tutorial covers how to use the modelscope ecosystem within OpenVINO.

## Notebook Contents
Throughout this notebook we will learn:
1. How to load a ModelScope pipeline then convert it with [OpenVINO Model Conversion API](https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model).
2. How to load transformers compatible models from ModelScope using OpenVINO integration with [Optimum Intel](https://github.com/huggingface/optimum-intel).
3. How to download and convert ModelScope Generative AI models using command-line interface and use them with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai).

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/modelscope-to-openvino/README.md" />
