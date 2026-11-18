# Moonshine Streaming automatic speech recognition with OpenVINO

[Moonshine Streaming Medium](https://huggingface.co/moonshine-ai/moonshine-streaming-medium) is a 245M-parameter English automatic speech recognition (ASR) model. It combines a sliding-window encoder with an autoregressive decoder and is designed for low-latency, on-device transcription.

This tutorial downloads the official Hugging Face checkpoint, optionally runs it with PyTorch XPU, converts it to a dynamic OpenVINO IR, and implements the growing decoder-prefix generation loop required by the exported model. It also demonstrates data-free NNCF weight-only INT8 and grouped INT4 compression.

## Notebook Contents

The tutorial covers:

1. License and evaluation-data prerequisites.
2. Package installation and source-model preparation.
3. Optional PyTorch XPU enablement and source-framework transcription.
4. Dynamic OpenVINO model export.
5. Growing-prefix OpenVINO inference.
6. CPU/GPU/NPU discovery, compilation, and performance-counter profiling.
7. Moonshine validation with optional LibriSpeech `test-clean` WER evaluation.
8. Data-free NNCF INT8 weight-only compression and validation.
9. Roofline interpretation and fusion observations.

### License and data

- The [Moonshine Streaming Medium checkpoint](https://huggingface.co/moonshine-ai/moonshine-streaming-medium) is published under the [MIT license](https://huggingface.co/moonshine-ai/moonshine-streaming-medium).
- The optional evaluation corpus is [LibriSpeech](https://www.openslr.org/12/), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Installation Instructions

This is a self-contained example. Run it in a Python virtual environment with Jupyter installed. The notebook installs its Python dependencies in the first code cell. For the general setup, see the [OpenVINO Notebooks Installation Guide](../../README.md).

The optional PyTorch XPU section requires an Intel XPU-enabled PyTorch build and a system with `torch.xpu.is_available()` set to `True`; otherwise it is skipped.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/moonshine-streaming-asr/README.md" />
