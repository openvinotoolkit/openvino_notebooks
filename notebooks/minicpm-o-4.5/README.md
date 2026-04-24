# MiniCPM-o 4.5 Multimodal Model with OpenVINO

<p align="center">
    <img src="https://raw.githubusercontent.com/OpenBMB/MiniCPM-o/main/assets/minicpm-o-45-framework.png" width="100%"/>
</p>

[MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) is the latest and most capable model in the MiniCPM-o series — an end-to-end omnimodal model with **9B parameters** built on **SigLip2 + Whisper-medium + CosyVoice2 + Qwen3-8B**. It achieves Gemini 2.5 Flash level performance on vision-language benchmarks with only 9B parameters.

## Key Features

- **Leading Visual Capability** — 77.6 on OpenCompass, surpassing GPT-4o and Gemini 2.0 Pro
- **Strong Speech Capability** — Bilingual (EN/ZH) real-time speech conversation with configurable voice cloning
- **Full-Duplex Live Streaming** — See, listen, and speak simultaneously with real-time video + audio
- **Proactive Interaction** — Initiates reminders/comments based on live scene understanding
- **Strong OCR** — State-of-the-art end-to-end English document parsing

## Architecture (16 Sub-models)

The model comprises 16 interconnected sub-models converted to OpenVINO IR format:

| Sub-model | Role | Quantization |
|-----------|------|:------------:|
| **LLM Embedding** | Token embeddings for Qwen3-8B backbone | — |
| **LLM Language Model** | Main language model with stateful KV cache | INT4 |
| **Vision Model** | SigLip2 vision encoder for image understanding | INT8 |
| **Resampler** | Projects vision features to LLM hidden space | — |
| **Audio Encoder** | Whisper-medium encoder for speech/audio understanding | — |
| **Audio Projection** | Projects audio features to LLM hidden space | — |
| **TTS Text Embedding** | Token embeddings for TTS LLaMA decoder | — |
| **TTS Language Model** | LLaMA decoder for speech token generation | — |
| **TTS Projector SPK** | Speaker embedding projector | — |
| **TTS Projector Semantic** | Semantic feature projector for TTS conditioning | — |
| **TTS Code Embedding** | Audio code token embeddings | — |
| **TTS Code Head** | Audio code prediction head | — |
| **Flow Embeddings** | CosyVoice2 flow-matching encoder + speaker projection | — |
| **Flow Encoder Chunk** | Streaming conformer encoder with KV cache | — |
| **Flow Estimator Chunk** | Unified DiT for flow-matching (streaming & non-streaming) | — |
| **HiFT** | Neural vocoder for mel-to-waveform synthesis | — |

> **Note:** The Flow Estimator Chunk model serves both streaming and non-streaming inference. In non-streaming mode it runs with empty KV caches (bit-identical to the legacy full estimator), while in streaming mode the KV caches enable temporally coherent cross-chunk mel generation — aligned with the original CosyVoice2 `flow.inference_chunk()` design. This unified approach saves ~220MB of memory.

## Notebook Contents

The notebook demonstrates:

1. **Prerequisites** — Install dependencies
2. **Convert & Quantize Model** — Export all 16 sub-models to OpenVINO IR with INT4/INT8 weight compression
3. **Select Inference Device** — Choose CPU, GPU, or NPU for different model components
4. **Run Inference** — Image understanding, audio understanding, omni-modal chat
5. **Interactive Demo** — Gradio-based multimodal chatbot

In this demonstration, you'll create an interactive chatbot that can answer questions about the provided image's content.

The image below illustrates example of input prompt and model answer.
![example.png](https://github.com/user-attachments/assets/906c5b2d-aa90-4d46-b417-9421b2061da2)

## Installation instructions
This is a self-contained example that relies solely on its own code.
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/minicpm-o-4.5/README.md" />
