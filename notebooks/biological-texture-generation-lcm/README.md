# Biological Texture Generation (Iris) using LCM & OpenVINO™

This notebook demonstrates how to generate high-fidelity biological textures, specifically human iris patterns, using **Latent Consistency Models (LCM)**.

## Key Features
- **Model:** [SimianLuo/LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)
- **Library:** [Optimum Intel](https://github.com/huggingface/optimum-intel) with OpenVINO™ backend.
- **Performance:** Generates high-quality images in just 4-8 steps (vs 25-50 steps for standard Stable Diffusion).
- **Hardware:** Supports Intel CPU, GPU (iGPU/dGPU), and NPU.

## Usage
1. Install dependencies.
2. Select your target device (CPU/GPU) from the widget.
3. Run the pipeline to generate synthetic iris data for biometric research.