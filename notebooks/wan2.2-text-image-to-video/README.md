# Text-Image to Video generation with Wan2.2 and OpenVINO

Wan2.2 is a comprehensive and open suite of video foundation models that pushes the boundaries of video generation. Wan2.2 is a major upgrade to Wan2.1 which includes following features:

- **Effective MoE Architecture**: Wan2.2 introduces a Mixture-of-Experts (MoE) architecture into video diffusion models. By separating the denoising process cross timesteps with specialized powerful expert models, this enlarges the overall model capacity while maintaining the same computational cost.

- **Cinematic-level Aesthetics**: Wan2.2 incorporates meticulously curated aesthetic data, complete with detailed labels for lighting, composition, contrast, color tone, and more. This allows for more precise and controllable cinematic style generation, facilitating the creation of videos with customizable aesthetic preferences.

- **Complex Motion Generation**: Compared to Wan2.1, Wan2.2 is trained on a significantly larger data, with +65.6% more images and +83.2% more videos. This expansion notably enhances the model's generalization across multiple dimensions such as motions, semantics, and aesthetics, achieving TOP performance among all open-sourced and closed-sourced models.

- **Efficient High-Definition Hybrid TI2V**: Wan2.2 open-sources a 5B model built with our advanced Wan2.2-VAE that achieves a compression ratio of 16×16×4. This model supports both text-to-video and image-to-video generation at 720P resolution with 24fps and can also run on consumer-grade graphics cards like 4090. It is one of the fastest 720P@24fps models currently available, capable of serving both the industrial and academic sectors simultaneously.

You can find more details about model in [model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) and [original repository](https://github.com/Wan-Video/Wan2.2)

<img width="962" height="1118" alt="image" src="https://github.com/user-attachments/assets/8bc4a9ca-9036-4efb-8738-4417db9f3164" />

## Notebook contents
This tutorial consists of the following steps:
- Prerequisites
- Convert and Optimize model
- Run inference pipeline
- Interactive inference

In this tutorial we consider how to convert, optimize and run Wan2.2 model for Text-Image to Video generation using OpenVINO.

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/wan2.2-text-image-to-video/README.md" />
