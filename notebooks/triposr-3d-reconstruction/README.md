# TripoSR feedforward 3D reconstruction from a single image and OpenVINO™


[TripoSR](https://huggingface.co/spaces/stabilityai/TripoSR) is a state-of-the-art open-source model for fast feedforward 3D reconstruction from a single image, developed in collaboration between [Tripo AI](https://www.tripo3d.ai/) and [Stability AI](https://stability.ai/news/triposr-3d-generation).

You can find [the source code on GitHub](https://github.com/VAST-AI-Research/TripoSR) and [demo on HuggingFace](https://huggingface.co/spaces/stabilityai/TripoSR). Also, you can read the paper [TripoSR: Fast 3D Object Reconstruction from a Single Image](https://arxiv.org/abs/2403.02151).


<div>
  <img src="https://github.com/VAST-AI-Research/TripoSR/blob/main/figures/teaser800.gif?raw=true" alt="Teaser Video">
</div>


## Notebook contents
The tutorial consists from following steps:

- Prerequisites
- Get the original model
- Convert the original model to OpenVINO Intermediate Representation (IR) format
- Compiling models and prepare pipeline
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/triposr-3d-reconstruction/README.md" />
