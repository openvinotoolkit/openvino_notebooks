# Automatic speech recognition using Distil-Whisper and OpenVINO

[Distil-Whisper](https://huggingface.co/distil-whisper/distil-large-v2) is a distilled variant of the [Whisper](https://huggingface.co/openai/whisper-large-v2) model by OpenAI proposed in the paper [Robust Knowledge Distillation via Large-Scale Pseudo Labelling](https://arxiv.org/abs/2311.00430). Compared to Whisper, Distil-Whisper runs in several times faster with 50% fewer parameters, while performing to within 1% word error rate (WER) on out-of-distribution evaluation data.

In this tutorial, we consider how to run Distil-Whisper using OpenVINO. We will use the pre-trained model from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library. To simplify the user experience, the [Hugging Face Optimum](https://huggingface.co/docs/optimum) library is used to convert the model to OpenVINO™ IR format. To further improve OpenVINO Distil-Whisper model performance `INT8` post-training quantization from [NNCF](https://github.com/openvinotoolkit/nncf/) is applied.

## Notebook Contents

This notebook demonstrates how to perform automatic speech recognition (ASR) using the Distil-Whisper model and OpenVINO.

The tutorial consists of following steps:
1. Download PyTorch model
2. Run PyTorch model inference
3. Convert and run the model using OpenVINO Integration with HuggingFace Optimum.
4. Compare the performance of PyTorch and the OpenVINO model.
5. Use the OpenVINO model with HuggingFace pipelines for long-form audio transcription.
6. Apply post-training quantization from [NNCF](https://github.com/openvinotoolkit/nncf/).
7. Launch an interactive demo for speech recognition


## Installation Instructions

This is a self-contained example that relies solely on its code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).