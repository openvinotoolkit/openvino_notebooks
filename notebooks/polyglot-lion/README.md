# Convert and Run Polyglot-Lion with OpenVINO

[Polyglot-Lion 1.7B](https://huggingface.co/knoveleng/polyglot-lion-1.7b) is a
multilingual speech-recognition model based on
[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).
This tutorial converts the original PyTorch model to stateful OpenVINO IR,
compares recognition accuracy on the same audio, and runs inference on Intel
CPU and optional GPU.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites and prepare FLEURS audio samples
- Run the original PyTorch model on CPU
- Convert the audio encoder and stateful decoder to OpenVINO IR
- Explore optional INT8 and mixed INT8/FP16 weight compression
- Compare transcripts, WER/CER and numerical outputs with PyTorch
- Benchmark complete-generation latency
- Run an interactive sample or local-WAV transcription demo

## Installation Instructions

We recommend running the notebook in a dedicated Python 3.12 virtual environment
on Linux. For details, refer to the [Installation Guide](../../README.md).
Open [polyglot-lion.ipynb](./polyglot-lion.ipynb), keep
[polyglot_helper.py](./polyglot_helper.py) beside it, and execute the cells in order.
The first cell activates a notebook-specific `.venv` for dependency isolation.
Enable `INSTALL_PACKAGES` for the setup cell, then restart the kernel and disable
it before continuing. Git is required for the pinned Optimum Intel dependency.
Allow about 16 GB for the checkpoint and all optional model variants, plus
environment and download caches.

**Experimental notebook:** the default run uses CPU and one English clip.
GPU inference, compression and multilingual evaluation are optional. The notebook
includes encoder and GPU precision corrections; full INT8 remains experimental
because it failed broader GPU accuracy testing. See the notebook's
[Conclusions](./polyglot-lion.ipynb#Conclusions) for validation results and limitations.

Polyglot-Lion declares MIT; Qwen3-ASR and Optimum Intel use Apache-2.0.
[FLEURS](https://huggingface.co/datasets/google/fleurs) is CC-BY-4.0
(Google / Conneau et al., 2022). Preserve applicable attribution and
[third-party notices](../../licensing/third-party-programs.txt).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/polyglot-lion/README.md" />
