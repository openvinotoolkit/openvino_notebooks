# Video Subtitle Generation with OpenAI Whisper
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/227-whisper-subtitles-generation/227-whisper-convert.ipynb)
[Whisper](https://openai.com/blog/whisper/) is a general-purpose speech recognition model from [OpenAI](https://openai.com/). The model is able to almost flawlessly transcribe speech across dozens of languages and even handle poor audio quality or excessive background noise.
This notebook will run the model with OpenVINO to generate transcription of a video.


## Notebook Contents

This notebook demonstrates how to generate video subtitles using the open-source Whisper model. Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web. It is a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.
You can find more information about this model in the [research paper](https://cdn.openai.com/papers/whisper.pdf), [OpenAI blog](https://openai.com/blog/whisper/), [model card](https://github.com/openai/whisper/blob/main/model-card.md) and GitHub [repository](https://github.com/openai/whisper).

This folder contains two notebooks that show how to convert and quantize model with OpenVINO:

1. [Convert Whisper model using OpenVINO](227-whisper-convert.ipynb)
2. [Quantize OpenVINO Whisper model using NNCF](227-whisper-nncf-quantize.ipynb)

In these notebooks, you will use its capabilities for generation of subtitles for a video.


### Convert Whisper model using OpenVINO
The first notebook contains the following steps:
1. Download the model.
2. Instantiate original PyTorch model pipeline.
3. Convert model to OpenVINO IR, using model conversion API.
4. Run the Whisper pipeline with OpenVINO.

A simplified demo pipeline is represented in the diagram below:
![whisper_pipeline.png](https://user-images.githubusercontent.com/29454499/204536733-1f4342f7-2328-476a-a431-cb596df69854.png)
The final output of running this notebook is an `srt file` (popular video captioning format) with subtitles for a sample video downloaded from YouTube.
This file can be integrated with a video player during playback or embedded directly into a video file with `ffmpeg` or similar tools that support working with subtitles.

The image below shows an example of the video as input and corresponding transcription as output.

![image](https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png)


### Quantize OpenVINO Whisper model using NNCF
The second notebook will guide you through steps of improving model performance by INT8 quantization with [NNCF](https://github.com/openvinotoolkit/nncf):
1. Quantize the converted OpenVINO model from [227-whisper-convert notebook](227-whisper-convert.ipynb) with NNCF.
2. Check model result for the demo video.
3. Compare model size, performance and accuracy of FP32 and quantized INT8 models.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
