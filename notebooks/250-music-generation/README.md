# Controllable Music Generation with MusicGen and OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F250-music-generation%2F250-music-generation.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/igor-davidyuk/openvino_notebooks/blob/musicgen/notebooks/250-music-generation/250-music-generation.ipynb)

MusicGen is a single-stage auto-regressive Transformer model capable of generating high-quality music samples conditioned on text descriptions or audio prompts. The text prompt is passed to a text encoder model (T5) to obtain a sequence of hidden-state representations. These hidden states are fed to MusicGen, which predicts discrete audio tokens (audio codes). Finally, audio tokens are then decoded using an audio compression model (EnCodec) to recover the audio waveform.

<img src="https://user-images.githubusercontent.com/76463150/260439306-81c81c8d-1f9c-41d0-b881-9491766def8e.png" width=700>

## Notebook Contents
The tutorial consists of the following steps:

- Install and import prerequisite packages
- Download the MusicGen Small model from a public source using the [Hugging Face Hub](https://huggingface.co/models?sort=downloads&search=facebook%2Fmusicgen-).
- Run the text-conditioned music generation pipeline
- Convert three models backing the MusicGen pipeline
- Run the music generation pipeline again using OpenVINO

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
