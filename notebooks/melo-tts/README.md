# MeloTTS Text-to-Speech with OpenVINO™

[MeloTTS](https://github.com/myshell-ai/MeloTTS) is a high-quality, multilingual text-to-speech (TTS) library developed by MyShell.ai, based on the [VITS](https://arxiv.org/abs/2106.06103) architecture. It supports English, Chinese (including mixed Chinese/English), Spanish, French, Japanese, and Korean, is fast enough for real-time inference on CPU, and produces natural, expressive speech.

This notebook demonstrates an end-to-end workflow for accelerating MeloTTS with OpenVINO: downloading the pretrained checkpoints, converting the PyTorch pipeline to OpenVINO Intermediate Representation (IR), and benchmarking the OpenVINO inference against the original PyTorch baseline.

To keep the conversion OpenVINO-friendly, the VITS pipeline is split into two statically-traceable sub-models, while the dynamic, data-dependent length regulator (`generate_path`) is kept in Python:

- `melotts_enc.xml` — speaker embedding + text encoder + duration predictors (`sdp` and `dp`)
- `melotts_dec.xml` — flow (reverse) + HiFiGAN generator

The BERT text frontend used for prosody can optionally also be converted to OpenVINO IR.

More details about the model can be found in the original [repository](https://github.com/myshell-ai/MeloTTS) and the pretrained models on [Hugging Face](https://huggingface.co/myshell-ai).

## Features

- **Multilingual** — English, Chinese (with Chinese/English code-switching), Spanish, French, Japanese, and Korean
- **VITS-based** — end-to-end synthesis with a stochastic duration predictor for natural prosody
- **Split for OpenVINO** — the network body is exported into two IR sub-models; dynamic length regulation stays in NumPy
- **Reproducible noise** — all random noise is passed in explicitly as model inputs, so results are deterministic
- **PyTorch vs OpenVINO benchmark** — reports total inference time, audio duration, and Real-Time Factor (RTF)
- **Self-contained** — all scripts and the `melo_torch` / `melo_openvino` packages are bundled in this folder, so it runs standalone

## Model Architecture

| Sub-model | Components | Role |
|-----------|------------|------|
| `melotts_enc.xml` | `emb_g` + `enc_p` (TextEncoder) + `sdp` + `dp` | Text encoding and duration / prior prediction |
| `melotts_dec.xml` | `flow` (reverse) + `dec` (HiFiGAN generator) | Latent → waveform synthesis |
| Length regulator | `generate_path` (Python / NumPy) | Data-dependent, dynamic-length expansion (kept outside IR) |
| BERT frontend (optional) | `hidden_states[-3]` of a masked-LM BERT | Text features for prosody |

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download checkpoints (MeloTTS + BERT)
- Convert the models to OpenVINO IR (encoder and decoder sub-models, plus optional BERT)
- Run inference and benchmark the PyTorch and OpenVINO pipelines (RTF comparison)
- Listen to the generated audio samples
- Launch an interactive Gradio demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a custom integration that may require environment-specific adjustments.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/melo-tts/README.md" />
