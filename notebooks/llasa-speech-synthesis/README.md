# Text-to-Speech synthesis using Llasa and OpenVINO

Llasa, is a text-to-speech (TTS) system that extends the text-based LLaMA (1B,3B, and 8B) language model by incorporating speech tokens from the [XCodec2](https://huggingface.co/HKUSTAudio/xcodec2) codebook, which contains 65 536 tokens.  The model is capable of generating speech either solely from input text or by utilizing a given speech prompt.
The method is seamlessly compatible with the Llama framework, making training TTS similar as training LLM (convert audios into single-codebook tokens and simply view it as a special language). It opens the possibility of existing method for compression, acceleration and finetuning for LLM to be applied. 

More details about model can be found in the [paper](https://arxiv.org/abs/2502.04128), [repository](https://github.com/zhenye234/LLaSA_training) and [model card](https://huggingface.co/HKUSTAudio/Llasa-3B).

In this tutorial we consider how to run Llasa pipeline using OpenVINO.

## Notebook Contents

The tutorial consists of the following steps:

* Convert the model to OpenVINO format using Optimum Intel
* Run Text-to-Speech synthesis using the OpenVINO model
* Interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llasa-speech-synthesis/README.md" />
