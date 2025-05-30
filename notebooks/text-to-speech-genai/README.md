# Text-to-Speech Generation with OpenVINO GenAI

[OpenVINO™ GenAI](https://github.com/openvinotoolkit/openvino.genai) is a library of the most popular Generative AI model pipelines, optimized execution methods, and samples that run on top of highly performant OpenVINO Runtime.

This library is friendly to PC and laptop execution, and optimized for resource consumption. It requires no external dependencies to run generative models as it already includes all the core functionality (e.g. tokenization via openvino-tokenizers).

Text-to-speech (TTS) technology converts written text into spoken audio. It's a form of speech synthesis that allows users to listen to digital text being read aloud. This technology is used in various applications, including accessibility for those with visual impairments, voice assistants, and language learning tools.

In this notebook we will demonstrate how to use OpenVINO GenAI capabilities for speech synthesis. You can find list of supported by OpenVINO GenAI in [SUPPORTED_MODELS.md](https://github.com/openvinotoolkit/openvino.genai/blob/master/SUPPORTED_MODELS.md#speech-generation-models). In this tutorial we will use [SpeechT5 TTS](https://huggingface.co/microsoft/speecht5_tts) model.

## Notebook Contents

The tutorial consists of the following steps:

* Convert the model to OpenVINO format using Optimum Intel
* Run Text-to-Speech synthesis using the OpenVINO model
* Run Text-to-Speech synthesis with Voice Cloning using the OpenVINO model
* Interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/text-to-speech-genai/README.md" />
