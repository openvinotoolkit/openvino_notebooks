# MMS: Scaling Speech Technology to 1000+ languages with OpenVINO™

The Massively Multilingual Speech (MMS) project expands speech technology from about 100 languages to over 1,000 by building a single multilingual speech recognition model supporting over 1,100 languages (more than 10 times as many as before), language identification models able to identify over 4,000 languages (40 times more than before), pretrained models supporting over 1,400 languages, and text-to-speech models for over 1,100 languages.
The MMS model was proposed in [Scaling Speech Technology to 1,000+ Languages](https://arxiv.org/abs/2305.13516).  The models and code are originally released [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms).
There are the different models open sourced in the MMS project: Automatic Speech Recognition (ASR), Language Identification (LID) and Speech Synthesis (TTS).


## Notebook Contents

This notebook demonstrates how to convert and run ASR and LID models using OpenVINO. An additional part demonstrates how to run models quantization with [NNCF](https://github.com/openvinotoolkit/nncf/) to improve their inference speed.

The tutorial consists of the following steps:

- Install and import prerequisite packages
- Download pretrained model and processor
- Prepare an example audio using [multilingual_librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) dataset
- Make inference with the original model
- Convert models to OpenVINO IR model and make inference
- Quantize models with [NNCF](https://github.com/openvinotoolkit/nncf/)
- Interactive demo with gradio

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).