# Synchronized audio-video generation with LTX-2.3 and OpenVINO(TM)

[LTX-2.3](https://huggingface.co/diffusers/LTX-2.3-Diffusers) is a diffusion transformer model that generates video and synchronized audio in a single pipeline. The prompt can describe both the visual scene and its soundtrack, including ambient sounds, music, and speech.

This tutorial shows how to export LTX-2.3 to OpenVINO Intermediate Representation with [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) and run one of two workflows:

- **Text-to-audio-video** generates video and audio from a text prompt.
- **Image-to-audio-video** uses an initial image together with a text prompt.

Only the selected workflow is exported because image-to-video additionally requires a VAE encoder and therefore has a different IR layout.

> **Resource notice:** LTX-2.3 is a large model. Download, export, and inference require substantial disk space and memory and may take a long time, especially on CPU.

## Notebook contents

- Prerequisites
- Select generation mode
- Convert the model to OpenVINO IR
- Run audio-video generation
- Interactive inference
- Limitations

## Installation instructions

This is a self-contained example that relies solely on its own code.

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/ltx2.3-audio-video/README.md" />
