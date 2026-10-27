# Multimodal AI with LFM3 and OpenVINO

[LFM3](https://www.liquid.ai/blog/lfm3) is a family of open-weight models from [Liquid AI](https://www.liquid.ai/) built on one shared backbone and designed for on-device use and domain-specific fine-tuning. The language model has a hybrid architecture that combines gated short convolutions with sliding-window and global attention layers.

The family covers several modalities:

| Model | Input | Output | Highlights |
|---|---|---|---|
| **LFM3-1.7B** | Text | Text | Reasoning on/off per request, tool calling, structured outputs, 256K context |
| **LFM3-VL** | Text, images, video | Text | New in-house vision encoder, video understanding, thinking mode |
| **LFM3-Omni** | Text, images, video, audio | Text, speech | First LFM omni model: speech recognition, text-to-speech, spoken chat |

## What The Notebook Covers

- Install prerequisites
- Download preconverted OpenVINO models or convert them with [Optimum Intel](https://github.com/huggingface/optimum-intel) (FP16 / INT8 / INT4)
- **LFM3-1.7B** with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) `LLMPipeline`: direct answers vs. reasoning, multi-turn chat, tool calling and structured JSON output
- **LFM3-VL** with Optimum Intel `OVModelForVisualCausalLM`: image understanding, multi-image comparison, thinking mode and video understanding
- **LFM3-Omni** with Optimum Intel `OVModelForMultimodalLM`: text-to-speech, speech recognition, spoken questions about images and spoken chat
- Interactive Gradio demo with LFM3-Omni
- Measure LFM3-1.7B performance with OpenVINO GenAI

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/lfm3/README.md" />
