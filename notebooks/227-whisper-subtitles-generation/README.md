# Video Subtitle Generation with OpenAI Whisper
[Whisper](https://openai.com/blog/whisper/) is a general-purpose speech recognition model from [OpenAI](https://openai.com/). The model is able to almost flawlessly transcribe speech across dozens of languages and even handle poor audio quality or excessive background noise.
This notebook will run the model with OpenVINO to generate transcription of a video.


## Notebook Contents

This notebook demonstrates how to generate video subtitles using the open-source Whisper model. Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web. It is a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.
You can find more information about this model in the [research paper](https://cdn.openai.com/papers/whisper.pdf), [OpenAI blog](https://openai.com/blog/whisper/), [model card](https://github.com/openai/whisper/blob/main/model-card.md) and GitHub [repository](https://github.com/openai/whisper).

In this notebook, you will use its capabilities for generation of subtitles for a video.
Notebook contains the following steps:
1. Download the model.
2. Instantiate original PyTorch model pipeline.
3. Export the ONNX model and convert it to OpenVINO IR, using the Model Optimizer tool.
4. Run the Whisper pipeline with OpenVINO.

A simplified demo pipeline is represented in the diagram below:
![whisper_pipeline.png](https://user-images.githubusercontent.com/29454499/204536733-1f4342f7-2328-476a-a431-cb596df69854.png)
The final output of running this notebook is an `srt file` (popular video captioning format) with subtitles for a sample video downloaded from YouTube.
This file can be integrated with a video player during playback or embedded directly into a video file with `ffmpeg` or similar tools that support working with subtitles.

The image below shows an example of the video as input and corresponding transcription as output.

![image](https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png)

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
