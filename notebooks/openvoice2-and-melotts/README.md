# Voice tone cloning with OpenVoice2 and MeloTTS for Text-to-Speech by OpenVINO



<!-- TODO: insert link with the image/gif -->
![sdf](https://github.com/openvinotoolkit/openvino_notebooks/assets/5703039/ca7eab80-148d-45b0-84e8-a5a279846b51)

OpenVoice2 is a versatile system for instant voice tone transferring and generating speech in various languages with just a brief audio snippet from the source speaker, using MeloTTS as the base speakers. OpenVoice2 includes all features from V1 and introduces several enhancements: (i) better audio quality: OpenVoice2 adopts a different training strategy that delivers superior audio quality. (ii) native multi-lingual support: English, Spanish, French, Chinese, Japanese, and Korean are natively supported. (iii) free commercial use: starting from April 2024, both V2 and V1 are released under the MIT License, allowing free commercial use.

OpenVoice2 retains the core strengths of OpenVoice1, including accurate tone color cloning, flexible voice style control, and zero-shot cross-lingual voice cloning.

More details about model can be found in [project web page](https://research.myshell.ai/open-voice), [paper](https://arxiv.org/abs/2312.01479), and official [repository](https://github.com/myshell-ai/OpenVoice)

In this tutorial we will explore how to convert and run OpenVoice2 and MeloTTS using OpenVINO.

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

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/openvoice2-and-melotts/README.md" />
