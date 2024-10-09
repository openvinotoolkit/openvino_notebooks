# Video Subtitle Generation with OpenAI Whisper and OpenVINO Generate API
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/whisper-subtitles-generation/whisper-subtitles-generation.ipynb)
[Whisper](https://openai.com/index/whisper/) is a general-purpose speech recognition model from [OpenAI](https://openai.com). The model is able to almost flawlessly transcribe speech across dozens of languages and even handle poor audio quality or excessive background noise.
This notebook will run the model with OpenVINO Generate API to generate transcription of a video.

## Notebook Contents

This notebook demonstrates how to generate video subtitles using the open-source Whisper model. Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web. It is a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.
You can find more information about this model in the [research paper](https://cdn.openai.com/papers/whisper.pdf), [OpenAI blog](https://openai.com/index/whisper/), [model card](https://github.com/openai/whisper/blob/main/model-card.md) and GitHub [repository](https://github.com/openai/whisper).

This folder contains notebook that show how to convert and quantize model with OpenVINO and run pipeline with [Generate API](https://github.com/openvinotoolkit/openvino.genai). We will use [NNCF](https://github.com/openvinotoolkit/nncf) improving model performance by INT8 quantization.

The notebook contains the following steps:
1. Download the model.
2. Instantiate original PyTorch model pipeline.
3. Convert model to OpenVINO IR, using model conversion API.
4. Run the Whisper pipeline with OpenVINO Generate API.
5. Quantize the OpenVINO model with NNCF.
6. Check quantized model result for the demo video.
7. Compare model size, performance and accuracy of FP32 and quantized INT8 models.
8. Launch Interactive demo for video subtitles generation.

In this notebook, you will use whisper capabilities for generation of subtitles for a video.
A simplified demo pipeline is represented in the diagram below:
![whisper_pipeline.png](https://user-images.githubusercontent.com/29454499/204536733-1f4342f7-2328-476a-a431-cb596df69854.png)
The final output of running this notebook is an `srt file` (popular video captioning format) with subtitles for a sample video downloaded from YouTube.
This file can be integrated with a video player during playback or embedded directly into a video file with `ffmpeg` or similar tools that support working with subtitles.

The image below shows an example of the video as input and corresponding transcription as output.

![image](https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png)


### Quantize OpenVINO Whisper model using NNCF
The second notebook will guide you through steps of  :


## Installation Instructions

This example requires `ffmpeg` to be installed. All other required dependencies will be installed by the notebook itself.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/whisper-subtitles-generation/README.md" />
