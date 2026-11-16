# Speech Understanding with SenseVoiceSmall and OpenVINO

[SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) is a non-autoregressive speech understanding model from FunAudioLLM. One encoder/CTC model produces a transcript, language identification and emotion/audio-event tags. The released Small checkpoint supports Mandarin, Cantonese, English, Japanese and Korean.

This tutorial exports the model with the Optimum Intel Python API and runs it with OpenVINO GenAI `ASRPipeline`. The `OVModelForSpeechSeq2Seq` export entry point also handles the SenseVoice CTC architecture. See the [original repository](https://github.com/FunAudioLLM/SenseVoice) and [paper](https://arxiv.org/abs/2407.04051).

This notebook offers CPU/GPU inference with FP16 or INT8 weights; NPU and INT4 are not included.

## Notebook contents

The tutorial consists of the following steps:

1. Install prerequisites
2. Export a revision-pinned model to FP16 or compress its weights to INT8 with NNCF
3. Transcribe short recordings and inspect language, emotion and event predictions
4. Select a language and compare inverse text normalization (ITN) settings
5. Transcribe two audio inputs in one batch
6. Launch an optional Gradio demo for uploaded files or microphone recordings

Recordings are limited to 30 seconds. Long-audio segmentation, translation, diarization and timestamps are outside this example's scope. Audio examples come from a small [LibriSpeech fixture](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy), derived from [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) (CC BY 4.0). Emotion/event tags are model predictions, not a classification accuracy assessment.

## Model License

The checkpoint uses the [custom FunASR model license](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE), not Apache-2.0. Review its terms before use. Export explicitly enables trusted remote code for a fixed source revision.

## Installation Instructions

This is a self-contained example that relies solely on its own code.

The notebook targets OpenVINO 2026.5. It installs pre-release OpenVINO/Tokenizers/GenAI packages from the nightly index with `--pre` and Optimum Intel from `main` for SenseVoice export, ASR, metadata and batching support. It uses compatible version ranges rather than fixing all package versions. Transformers is limited to 4.57.x, Gradio to 5.x, and matching CPU PyTorch/TorchAudio 2.8.x wheels are used for export. Dependencies are resolved together to preserve compatibility constraints. OpenVINO GPU inference does not require PyTorch GPU wheels.

The `protobuf` lower bound keeps its runtime compatible with the generated `sentencepiece` model definitions used for detokenizer conversion.

GPU inference explicitly requests FP32 runtime precision to avoid empty transcripts with the FP16 graph. Weight compression and runtime inference precision are separate settings.

The source checkpoint is approximately 0.94 GB, plus exported models and installed packages. A microphone is optional; file uploads work without one. Standard audio formats supported by SoundFile do not require a separate system FFmpeg installation.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to the [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/sensevoice-small/README.md" />
