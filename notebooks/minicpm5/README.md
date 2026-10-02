# Chat with MiniCPM5-2B and OpenVINO

[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) is a dense 2B-parameter language model from OpenBMB's MiniCPM5 series, designed for on-device deployment and resource-constrained scenarios. It reaches 2B-class open-source SOTA while staying competitive with 4B-class models, with advantages in coding, mathematics, long-context understanding, tool use and agentic tasks.

**Key Features of MiniCPM5-2B:**

- **Hybrid Reasoning** — Produces a `<think>...</think>` reasoning trace by default; thinking can be disabled per-request via the `enable_thinking` chat-template flag for fast, direct answers.
- **2B-Class SOTA Quality** — Post-trained with SFT, RL and On-Policy Distillation over the open [UltraData](https://ultradata.openbmb.cn/) corpus.
- **Long Context** — Native support for up to 131,072 tokens.
- **Efficient Architecture** — `LlamaForCausalLM` backbone with 42 layers and Grouped Query Attention (16 query heads, 2 KV heads).
- **Tool Calling** — Native function-calling chat template for agentic workflows.

More details can be found in the [model card](https://huggingface.co/openbmb/MiniCPM5-2B), the [MiniCPM tech report](https://arxiv.org/pdf/2506.07900) and the [GitHub repository](https://github.com/OpenBMB/MiniCPM).

In this tutorial we convert and compress the model using [Optimum Intel](https://github.com/huggingface/optimum-intel) with [NNCF](https://github.com/openvinotoolkit/nncf) weight compression, run text generation with the OpenVINO GenAI `LLMPipeline` on CPU and GPU, toggle the model's thinking mode, and launch an interactive Gradio chat. The NPU device is not supported in this example.

### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Select the weight format for export
- Convert and optimize the model using Optimum Intel CLI
- Create an OpenVINO GenAI inference pipeline
- Chat with the model in thinking and fast (non-thinking) modes
- Launch an interactive multi-turn Gradio demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/minicpm5/README.md" />
