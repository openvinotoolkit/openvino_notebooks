# Voice tone cloning with OpenVoice and OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2F284-openvoice%2F284-openvoice.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/284-openvoice/284-openvoice.ipynb)

<!-- TODO: insert link with the image/gif -->
![sdf](https://github.com/openvinotoolkit/openvino_notebooks/assets/5703039/ca7eab80-148d-45b0-84e8-a5a279846b51)

[OpenVoice](https://github.com/myshell-ai/OpenVoice) a versatile instant voice tone transferring and generating speech in various languages with just a brief audio snippet from the source speaker. OpenVoice represents has three main features: (i) high quality tone color replication with multiple languages and accents; (ii) it provides fine-tuned control over voice styles, including emotions, accents, as well as other parameters such as rhythm, pauses, and intonation. (iii) OpenVoice achieves zero-shot cross-lingual voice cloning, eliminating the need for the generated speech and the reference speech to be part of a massive-speaker multilingual training dataset

More details about model can be found in [project web page](https://research.myshell.ai/open-voice), [paper](https://arxiv.org/abs/2312.01479), and official [repository](https://github.com/myshell-ai/OpenVoice)

In this tutorial we will explore how to convert and run OpenVoice using OpenVINO.

## Notebook Contents

This notebook demonstrates voice tone cloning with [OpenVoice](https://github.com/myshell-ai/OpenVoice) in OpenVINO.

The tutorial consists of following steps:
- Install prerequisites
- Load PyTorch model
- Convert Model to Openvino Intermediate Representation format
- Run OpenVINO model inference on a single example
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
