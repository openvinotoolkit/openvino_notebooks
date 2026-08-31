# Visual-language assistant with Ministral-3 and OpenVINO

Ministral-3 is a family of compact multimodal models from Mistral AI that combines a language model with a Pixtral vision encoder. This tutorial supports the 3B and 8B instruction-following checkpoints and the `Ministral-3-3B-Reasoning-2512` model.

**Key Features of Ministral-3:**

- **Multimodal Understanding** — Combines text and vision capabilities in a compact 3B parameter model, enabling image understanding and visual question answering.
- **Reasoning** — The Reasoning checkpoint produces a `[THINK]...[/THINK]` trace before its final answer. The demo displays this trace separately and preserves it in multi-turn context.
- **Long Context Support** — Supports up to 262,144 tokens with YaRN RoPE scaling for extended context processing.
- **Efficient Architecture** — Uses Grouped Query Attention (32 attention heads with 8 KV heads) for memory-efficient inference.
- **Pixtral Vision Encoder** — Employs a PixtralVisionModel with patch-based image processing and multi-modal projection for seamless vision-language integration.

More details can be found in the [Instruct model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16), the [Reasoning model card](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512), and the [Mistral AI documentation](https://docs.mistral.ai/).

In this tutorial we consider how to convert and optimize Ministral-3 model for creating a multimodal chatbot. We use [Optimum Intel](https://github.com/huggingface/optimum-intel) for model conversion with [NNCF](https://github.com/openvinotoolkit/nncf) weight compression, and `OVModelForVisualCausalLM` for efficient inference with OpenVINO.

The `-BF16` suffix identifies the source checkpoint precision. The Instruct checkpoint without this suffix is FP8, while the Reasoning checkpoint is BF16 despite having no precision suffix. The notebook's FP16/INT4 selector controls the exported OpenVINO precision independently of the source checkpoint.

### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Select model and weight format
- Convert and optimize model using Optimum Intel CLI
- Run model inference with OpenVINO
- Separate the Reasoning model's reasoning trace from its final answer
- Launch an interactive multi-turn Gradio demo

In this demonstration, you'll create an interactive chatbot that can answer questions about provided image content.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

The merged Optimum Intel PR #1627 supports the outer `mistral3` architecture, but Ministral-3 checkpoints use the newer internal language-model type `ministral3`. This notebook temporarily pins Optimum Intel PR #1659, which adds that missing export path. Video, audio, and NPU inference are not supported in this example.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/ministral-3/README.md" />