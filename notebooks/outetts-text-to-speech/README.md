# Text-to-Speech synthesis using OuteTTS and OpenVINO

[OuteTTS-0.1-350M](https://huggingface.co/OuteAI/OuteTTS-0.1-350M) is a novel text-to-speech synthesis model that leverages pure language modeling without external adapters or complex architectures, built upon the LLaMa architecture. It demonstrates that high-quality speech synthesis is achievable through a straightforward approach using crafted prompts and audio tokens.

More details about model can be found in [original repo](https://github.com/edwko/OuteTTS).

In this tutorial we consider how to run OuteTTS pipeline using OpenVINO.

## Notebook Contents

The tutorial consists of the following steps:

* Convert model to OpenVINO format using Optimum Intel
* Run Text-to-Speech synthesis using OpenVINO model
* Run Text-to-Speech synthesis with Voice Cloning using OpenVINO model
* Interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/outetts-text-to-speech/README.md" />
