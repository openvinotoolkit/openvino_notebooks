# PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis with OpenVINO™

[This paper](https://arxiv.org/abs/2310.00426) introduces [PIXART-α](https://github.com/PixArt-alpha/PixArt-alpha), a Transformer-based T2I diffusion model whose image generation quality is competitive with state-of-the-art image generators, reaching near-commercial application standards. Additionally, it supports high-resolution image synthesis up to 1024px resolution with low training cost. To achieve this goal, three core designs are proposed: 
1. Training strategy decomposition: We devise three distinct training steps that separately optimize pixel dependency, text-image alignment, and image aesthetic quality; 
2. Efficient T2I Transformer: We incorporate cross-attention modules into Diffusion Transformer (DiT) to inject text conditions and streamline the computation-intensive class-condition branch; 
3. High-informative data: We emphasize the significance of concept density in text-image pairs and leverage a large Vision-Language model to auto-label dense pseudo-captions to assist text-image alignment learning. 

![](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/resolve/main/asset/images/teaser.png)

## Notebook Contents

This notebook demonstrates how to convert and run the Paint-by-Example model using OpenVINO. An additional part demonstrates how to run optimization with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up pipeline.

Notebook contains the following steps:
1. Convert PyTorch models to OpenVINO IR format.
2. Run PixArt-α pipeline with OpenVINO.
3. Optimize pipeline with [NNCF](https://github.com/openvinotoolkit/nncf/)
4. Compare results of FP16 and optimized pipelines
5. Interactive demo.

## Installation instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/pixart/README.md" />
