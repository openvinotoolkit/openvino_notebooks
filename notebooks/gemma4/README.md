# Visual-language assistant with Gemma 4 and OpenVINO

![](https://ai.google.dev/gemma/images/gemma4_banner.png)

[Gemma 4](https://huggingface.co/collections/google/gemma-4) is Google DeepMind's latest generation of open multimodal models. It handles text and image input and generates text output.

The family includes four sizes:

| Model | Total Params | Active Params | Modalities | Architecture |
|---|---|---|---|---|
| **E2B** | 5.1 B | 2.3 B | Text, Image, Audio | Dense + PLE |
| **E4B** | 8 B | 4.5 B | Text, Image, Audio | Dense + PLE |
| **26B‑A4B** | 25.2 B | 3.8 B | Text, Image | MoE |
| **31B** | 30.7 B | 30.7 B | Text, Image | Dense |

Key features: built-in chain-of-thought (thinking mode), native system prompt support, interleaved multi-image input, and MoE architecture for efficient inference.


## What The Notebook Covers

- Install prerequisites
- Select a Gemma 4 model and target weight format (FP16 / INT8 / INT4)
- Convert the model to OpenVINO IR using Optimum Intel
- Run image-understanding inference via `OVModelForVisualCausalLM`
- Demonstrate native system prompt support
- Demonstrate interleaved multi-image input
- Demonstrate thinking mode (chain-of-thought reasoning)
- Launch an interactive Gradio chat demo with image and video upload

## Installation Instructions

This is a self-contained example that relies on the notebook-local helper code. We recommend running it in a dedicated virtual environment with Jupyter available.

For general environment setup, see the main [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/gemma4/README.md" />
