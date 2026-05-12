# Automatic Speech Recognition using Whisper and OpenVINO with Generate API

[Whisper](https://openai.com/index/whisper/) is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web.

In this tutorial, we consider how to run Whisper using OpenVINO with Generate API. We use pre-converted models from the [OpenVINO collection on HuggingFace](https://huggingface.co/collections/OpenVINO/speech-to-text) or convert models locally using [Hugging Face Optimum Intel](https://huggingface.co/docs/optimum/intel/index). To simplify the user experience, we use [OpenVINO Generate API](https://github.com/openvinotoolkit/openvino.genai) for [Whisper automatic speech recognition scenarios](https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/whisper_speech_recognition/README.md).

## Notebook Contents

This notebook demonstrates how to perform automatic speech recognition (ASR) and video subtitle generation using the Whisper model and OpenVINO.

The notebook supports:
- **Whisper** multilingual models (large-v3-turbo, large-v3, large-v2, medium, small, base)
- **Distil-Whisper** English-only models (distil-large-v3, distil-large-v2, distil-medium.en, distil-small.en)
- **Pre-converted models** from the [OpenVINO HuggingFace collection](https://huggingface.co/collections/OpenVINO/speech-to-text) in FP16, INT8, and INT4 precisions
- **Audio transcription** and **multilingual speech translation**
- **Video subtitle generation** with `.srt` output

The tutorial consists of the following steps:
1. Select and download a pre-converted OpenVINO model (or convert locally)
2. Run the Whisper pipeline with OpenVINO Generate API
3. Perform audio transcription with timestamps
4. Perform multilingual speech translation
5. Generate video subtitles in SRT format
6. Launch an interactive Gradio demo with Audio and Video tabs

## Installation Instructions

This example requires `ffmpeg` to be installed. All other required dependencies will be installed by the notebook itself.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/whisper-asr-genai/README.md" />