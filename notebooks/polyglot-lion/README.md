# Convert and Run Polyglot-Lion with OpenVINO

[Polyglot-Lion](https://huggingface.co/knoveleng/polyglot-lion-0.6b) is a
multilingual speech-recognition model based on
[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).
This tutorial converts the original PyTorch model to stateful OpenVINO IR,
compares recognition accuracy on the same audio, and runs inference on Intel
CPU and optional GPU. The compact 0.6B model is the default; the larger
[1.7B model](https://huggingface.co/knoveleng/polyglot-lion-1.7b) is optional.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites and prepare FLEURS audio samples
- Export the stateful OpenVINO model
- Run the original PyTorch model on CPU and re-export the audio encoder
- Explore optional INT8 and mixed INT8/FP16 weight compression
- Compare transcripts, WER/CER and numerical outputs with PyTorch
- Benchmark complete-generation latency
- Optionally compare PyTorch XPU transcription, logits and latency
- Run an interactive sample or local-WAV transcription demo

## Installation Instructions

Use Python 3.11–3.13 with pip and Git. Open
[polyglot-lion.ipynb](./polyglot-lion.ipynb) and execute the cells in order.
The first cell activates a notebook-specific `.venv`; dependencies install
automatically. When testing a pull-request branch, open the notebook from a checkout or
keep the matching `polyglot_helper.py` beside it. Plan for about 8 GB of system
RAM for the default 0.6B model, or 16 GB for the optional 1.7B model. These
are planning estimates, not measured peaks. The model weights occupy about 1.58 GB or
4.08 GB, respectively, in addition to the environment, converted IR and caches.
Local WAV files longer than 60 seconds are rejected to bound full-context
encoder memory use.

For the optional PyTorch XPU comparison, run the setup and installation cells,
install compatible XPU-enabled PyTorch and TorchAudio builds in the notebook's
`.venv`, restart the kernel, and set `RUN_PYTORCH_XPU = True`.
See the notebook for installation guidance.
It skips when XPU is unavailable and does not change the default CPU reference
or require XPU for OpenVINO GPU inference. The XPU model temporarily shares GPU
memory with OpenVINO; available memory and runtime support are device-specific.

**Experimental notebook:** the default run uses CPU and one English clip.
GPU inference, compression and multilingual evaluation are optional. The notebook
includes encoder and GPU precision corrections; full INT8 remains experimental.
An earlier 1.7B GPU full-INT8 evaluation exceeded the Mandarin accuracy budget.
The 1.7B option remains pinned to the checkpoint used for the encoder and
compression validation; newer checkpoint releases are not substituted without
rechecking those results. See the notebook's
[Conclusions](./polyglot-lion.ipynb#Conclusions) for results and limitations.

Polyglot-Lion declares MIT; Qwen3-ASR and Optimum Intel use Apache-2.0.
[FLEURS](https://huggingface.co/datasets/google/fleurs) is CC-BY-4.0
(Google / Conneau et al., 2022). Preserve applicable attribution and
[third-party notices](../../licensing/third-party-programs.txt).
See the [Polyglot-Lion paper](https://arxiv.org/abs/2603.16184) and the
[model project](https://github.com/knoveleng/polyglot-lion) for model details.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/polyglot-lion/README.md" />
