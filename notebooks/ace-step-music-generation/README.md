# Music generation using ACE Step and OpenVINO

[ACE-Step](https://ace-step.github.io/) is a novel open-source foundation model for music generation that overcomes key limitations of existing approaches and achieves state-of-the-art performance through a holistic architectural design. Current methods face inherent trade-offs between generation speed, musical coherence, and controllability. ACE-Step bridges this gap by integrating diffusion-based generation with Sana’s Deep Compression AutoEncoder (DCAE) and a lightweight linear transformer. The model achieving superior musical coherence and lyric alignment across melody, harmony, and rhythm metrics. Moreover, ACE-Step preserves fine-grained acoustic details, enabling advanced control mechanisms such as voice cloning, lyric editing, remixing, and track generation (e.g., lyric2vocal, singing2accompaniment). 

ACE-Step adapts a text-to-image diffusion framework for music generation. The core generative model is a diffusion model operating on a compressed mel spectrogram latent representation. This process is guided by conditioning information from three specialized encoders: a text prompt encoder, a lyric encoder, and a speaker encoder. Embeddings from these encoders are concatenated and integrated into the diffusion model via cross-attention mechanisms

ACE-Step can be used for generating original music from text descriptions, music remixing and style transfer, edit song lyrics. The model offers a set of controllable features that allow users to precisely control the generation process and enable targeted modifications to existing audio material, as well as perform specialized generation tasks through fine-tuning.

<img src="https://raw.githubusercontent.com/ACE-Step/ACE-Step/main/assets/ACE-Step_framework.png" width=90% style="display: block; margin: auto;" />

More details about the model can be found using the following resources: [project page](https://ace-step.github.io/), [paper](https://arxiv.org/abs/2506.00045), [original repository](https://github.com/ace-step/ACE-Step).


## Notebook Contents

This notebook demonstrates how to convert and run music generation or editing with ACE Step using OpenVINO.

The tutorial consists of the following steps:

- Install prerequisites
- Download and run inference of ACE Step model
- Convert the model to IR format and run inference with OpenVINO
- Download, apply and generate audio with LoRA
- Interactive demo


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/ace-step-music-generation/README.md" />
