# Automatic speech recognition using Whisper and OpenVINO with Generate API

[Whisper](https://openai.com/index/whisper/) is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web.

In this tutorial, we consider how to run Whisper using OpenVINO with Generate API. We will use the pre-trained model from the [Hugging Face Transformers](https://github.com/openvinotoolkit/openvino.genai) library. The [Hugging Face Optimum Intel](https://huggingface.co/docs/optimum/intel/index) library converts the models to OpenVINO™ IR format. To simplify the user experience, we will use [OpenVINO Generate API](https://github.com/openvinotoolkit/openvino.genai) for [Whisper automatic speech recognition scenarios](https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/whisper_speech_recognition/README.md).

## Notebook Contents

This notebook demonstrates how to perform automatic speech recognition (ASR) using the Whisper model and OpenVINO.

The tutorial consists of following steps:
1. Download PyTorch model
2. Run PyTorch model inference
3. Convert the model using OpenVINO Integration with HuggingFace Optimum.
4. Run the model using Generate API.
5. Compare the performance of PyTorch and the OpenVINO model.
6. Quantize the OpenVINO model with NNCF.
7. Check quantized model result for the demo video.
8. Compare model size, performance and accuracy of original and quantized models.
9. Launch an interactive demo for speech recognition


## Installation Instructions

This example requires `ffmpeg` to be installed. All other required dependencies will be installed by the notebook itself.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/whisper-asr-genai/README.md" />