# Image generation with Sana and OpenVINO

Sana is a text-to-image framework that can efficiently generate images up to 4096 × 4096 resolution developed by NVLabs. Sana can synthesize high-resolution, high-quality images with strong text-image alignment at a remarkably fast speed, deployable on laptop GPU. 
Core designs include: 
* **Deep compression autoencoder**: unlike traditional AEs, which compress images only 8×, we trained an AE that can compress images 32×, effectively reducing the number of latent tokens.
* **Linear DiT**: authors replaced all vanilla attention in DiT with linear attention, which is more efficient at high resolutions without sacrificing quality.
* **Decoder-only text encoder***: T5 replaced by modern decoder-only small LLM as the text encoder and designed complex human instruction with in-context learning to enhance the image-text alignment.
* **Efficient training and sampling**: Proposed Flow-DPM-Solver to reduce sampling steps, with efficient caption labeling and selection to accelerate convergence.

More details about model can be found in [paper](https://arxiv.org/abs/2410.10629), [model page](https://nvlabs.github.io/Sana/) and [original repo](https://github.com/NVlabs/Sana)
In this tutorial, we consider how to run Sana model using OpenVINO.

### Notebook Contents

In this demonstration, you will learn how to perform text-to-image generation using Sana and OpenVINO. 

Example of model work:

**Input prompt**: *a cyberpunk cat with a neon sign that says "Sana"*
![](https://github.com/user-attachments/assets/bacfcd2a-ac36-4421-9d1b-4e34aa0a9f62)

The tutorial consists of the following steps:

- Install prerequisites
- Collect Pytorch model pipeline
- Convert model to OpenVINO intermediate representation (IR) format
- Compress model weights using NNCF
- Prepare OpenVINO Inference pipeline
- Run Text-to-Image generation
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/sana-image-generation/README.md" />
