# pyannote Speaker Diarization and Embedding with OpenVINO™

This folder contains two notebooks that enable [pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker models with OpenVINO:

| Notebook | Task | Model |
|---|---|---|
| [pyannote-audio.ipynb](pyannote-audio.ipynb) | Speaker **diarization** (*"who spoke when?"*) | [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) |
| [pyannote-embedding.ipynb](pyannote-embedding.ipynb) | Speaker **embedding** / verification | [`pyannote/embedding`](https://huggingface.co/pyannote/embedding) |

> **Note:** both models are gated. Before running a notebook, accept the user conditions of the models it uses on Hugging Face and log in with a *Read* access token. The diarization notebook additionally uses [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0).

## Speaker diarization

Speaker diarization answers the question *"who spoke when?"* by partitioning a recording into speech segments and assigning each segment to a speaker, without knowing the speakers in advance. It is a common building block for meeting transcription, call-center analytics, and preparing training data for speech models.

The [pyannote-audio.ipynb](pyannote-audio.ipynb) notebook uses the open-source [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) pipeline and accelerates it with OpenVINO. It is a modern replacement for the speaker-diarization notebook that was deprecated in 2024, based on the newer `community-1` pipeline.

The pipeline has three stages:

- **segmentation** ([`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0), a PyanNet model) — detects speech and overlapped speech,
- **speaker embedding** (a WeSpeaker ResNet34 model) — turns speech into speaker vectors,
- **clustering** — groups the vectors into speakers.

The two heavy neural blocks (segmentation and the embedding ResNet) are converted to OpenVINO IR and run on the OpenVINO device you select (CPU or GPU), while pyannote keeps orchestrating windowing and clustering in Python.

The tutorial consists of the following steps:

- Install prerequisites and authenticate with Hugging Face
- Load the diarization pipeline and run a PyTorch baseline on a sample audio file
- Convert the segmentation and speaker-embedding blocks to OpenVINO IR
- Select an inference device and run the OpenVINO-accelerated pipeline on CPU or GPU
- *(Optional)* Run the pipeline on an Intel GPU through PyTorch's XPU backend
- *(Optional)* Measure the Diarization Error Rate (DER) on the [VoxConverse](https://github.com/joonson/voxconverse) test set

## Speaker embedding

Speaker embedding turns a speech recording into a fixed-size vector that captures *who* is speaking, independent of *what* is said. Two clips from the same speaker have a small cosine distance, while clips from different speakers are far apart — the core building block of speaker verification and comparison.

The [pyannote-embedding.ipynb](pyannote-embedding.ipynb) notebook uses the open-source [`pyannote/embedding`](https://huggingface.co/pyannote/embedding) model (an x-vector TDNN network with a trainable SincNet front-end that maps a 16 kHz waveform to a 512-dimensional speaker vector) and accelerates it with OpenVINO. Its model card reports a 2.8% Equal Error Rate (EER) on the VoxCeleb1 test set using cosine distance directly.

The tutorial consists of the following steps:

- Install prerequisites and authenticate with Hugging Face
- Load the embedding model and extract PyTorch baseline embeddings from sample clips
- Convert the model to OpenVINO IR
- Select an inference device and extract embeddings with OpenVINO, comparing against the PyTorch baseline
- *(Optional)* Run the model on an Intel GPU through PyTorch's XPU backend
- *(Optional)* Reproduce the VoxCeleb1 speaker-verification EER

In both notebooks the optional Intel XPU and full benchmark sections are disabled by default and are skipped automatically on machines that do not have the required hardware, so the notebooks stay cross-platform and CI-friendly.

## Installation instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/pyannote-audio/README.md" />
