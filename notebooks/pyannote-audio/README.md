# Speaker Diarization with pyannote and OpenVINO™

Speaker diarization answers the question *"who spoke when?"* by partitioning a recording into speech segments and assigning each segment to a speaker, without knowing the speakers in advance. It is a common building block for meeting transcription, call-center analytics, and preparing training data for speech models.

This notebook uses the open-source [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) pipeline and accelerates it with OpenVINO. It is a modern replacement for the speaker-diarization notebook that was deprecated in 2024, based on the newer `community-1` pipeline.

The pipeline has three stages:

- **segmentation** ([`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0), a PyanNet model) — detects speech and overlapped speech,
- **speaker embedding** (a WeSpeaker ResNet34 model) — turns speech into speaker vectors,
- **clustering** — groups the vectors into speakers.

The two heavy neural blocks (segmentation and the embedding ResNet) are converted to OpenVINO IR and run on the OpenVINO device you select (CPU or GPU), while pyannote keeps orchestrating windowing and clustering in Python.

> **Note:** `pyannote/speaker-diarization-community-1` is a gated model. Before running the notebook, accept the user conditions for both [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) and [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) on Hugging Face and log in with a *Read* access token.

## Notebook Contents

The [pyannote-audio](pyannote-audio.ipynb) notebook demonstrates the end-to-end enablement flow for the pyannote diarization pipeline with OpenVINO.

The tutorial consists of the following steps:

- Install prerequisites and authenticate with Hugging Face
- Load the diarization pipeline and run a PyTorch baseline on a sample audio file
- Convert the segmentation and speaker-embedding blocks to OpenVINO IR
- Select an inference device and run the OpenVINO-accelerated pipeline on CPU or GPU
- *(Optional)* Run the pipeline on an Intel GPU through PyTorch's XPU backend
- *(Optional)* Measure the Diarization Error Rate (DER) on the [VoxConverse](https://github.com/joonson/voxconverse) test set

The optional Intel XPU and full VoxConverse benchmark sections are disabled by default and are skipped automatically on machines that do not have the required hardware, so the notebook stays cross-platform and CI-friendly.

## Installation instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/pyannote-audio/README.md" />
