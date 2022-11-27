# Video subtitles generation with Whisper


## Notebook Contents

This notebook demonstrates how to generate video subtiteles using Whisper model. Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web.  It is a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.
You can find more information about this model in [paper](https://cdn.openai.com/papers/whisper.pdf), [OpenAI blogpost](https://openai.com/blog/whisper/), [model card](https://github.com/openai/whisper/blob/main/model-card.md) and [repository](https://github.com/openai/whisper).

In this notebook we will use its capabilities for generation subtitles to video.
Notebook contains following steps:
1. Download model.
2. Instantiate original model pipeline
3. Convert model to ONNX and then to IR using OpenVINO Model Optimizer tool.
4. Run Whisper pipeline with OpenVINO models.

The result of demo work is `srt file` (one of the most popular video captioning format) with subtitles for video downloaded from YouTube video hosting. This file can be integrated to player during video playback or directly embedded to video file using `ffmpeg` or any other tools for working with subtitles. 

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).