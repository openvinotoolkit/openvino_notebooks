# VoxCPM2 Text-to-Speech with OpenVINO™

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) is a 2-billion-parameter tokenizer-free diffusion autoregressive TTS model from OpenBMB.
It generates highly realistic multilingual speech and supports voice cloning and voice design via natural language descriptions.

## Features

- **Tokenizer-free** — operates directly on continuous audio latents (no discrete codec)
- **Multilingual** — input text in any of the 30 supported languages and synthesize directly, no language tag needed
- **Voice Design** — describe the desired voice in natural language
- **Voice Cloning** — clone a voice from a reference audio clip
- **Ultimate Cloning** — transcript-guided audio continuation for faithful reproduction

## Tutorial Objectives

1. Install required dependencies
2. Convert all VoxCPM2 sub-models to OpenVINO format (8 models)
3. Create a pure-OpenVINO inference pipeline independent of PyTorch
4. Build an interactive Gradio demo for text-to-speech

## Model Architecture

| Component | Description |
|-----------|-------------|
| AudioVAE Encoder | Compresses raw audio to 64-dim latent (640× down sampling) |
| Local Encoder | Encodes latent patches via 12-layer MiniCPM (non-causal) |
| Base LM | 28-layer MiniCPM4 autoregressive language model |
| Residual LM | 8-layer MiniCPM4 refinement model (no RoPE) |
| LocDiT | 12-layer diffusion transformer with Euler flow matching |
| AudioVAE Decoder | Decodes latent to 48 kHz waveform (960× upsampling) |

### Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/voxcpm2-tts/README.md" />
