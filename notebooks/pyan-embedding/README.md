# pyannote/embedding OpenVINO Notebook

This folder contains the notebook workflow for enabling the Hugging Face [`pyannote/embedding`](https://huggingface.co/pyannote/embedding) speaker embedding model with OpenVINO on CPU and GPU.

## Model Overview

`pyannote/embedding` is a speaker embedding model from the `pyannote.audio` project. It converts speech audio into a fixed-size speaker vector that can be used for speaker verification, speaker comparison, and speaker diarization pipelines.

The model is based on an x-vector TDNN-style speaker embedding architecture with trainable SincNet-style front-end features. In normal PyTorch usage, the model can be loaded with `pyannote.audio` and used to extract embeddings from complete audio files, excerpts, or sliding windows.

The Hugging Face model card reports 2.8% Equal Error Rate (EER) on the VoxCeleb1 test set when using cosine distance directly, without VAD or PLDA post-processing.

Access note: this is a gated Hugging Face model. Before running conversion or inference, accept the model terms at https://huggingface.co/pyannote/embedding and log in with a Hugging Face access token.

## What This Notebook Represents

[`inference.ipynb`](inference.ipynb) represents the OpenVINO enablement flow for this model:

1. Create and select the `ov_pyan` Conda environment.
2. Authenticate with Hugging Face.
3. Convert the PyTorch `pyannote/embedding` model to OpenVINO IR.
4. Run sample speaker embedding inference on OpenVINO CPU and GPU.
5. Download the VoxCeleb1 verification data.
6. Run the VoxCeleb1 EER benchmark on OpenVINO CPU and GPU.

The notebook is configured to keep generated artifacts inside this `notebook/` folder.

## Purpose

This notebook is intended to demonstrate that `pyannote/embedding` can be converted from PyTorch to OpenVINO IR and executed on OpenVINO CPU/GPU devices for speaker embedding extraction and VoxCeleb1 EER benchmarking.
