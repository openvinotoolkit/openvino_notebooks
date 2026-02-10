#  Text-to-Speech (TTS) system with Fun-CosyVoice 3.0 and OpenVINO

Fun-CosyVoice 3.0 is an advanced text-to-speech (TTS) system based on large language models (LLM), surpassing its predecessor (CosyVoice 2.0) in content consistency, speaker similarity, and prosody naturalness. It is designed for zero-shot multilingual speech synthesis in the wild.

More details can be found in the original [repository](https://github.com/FunAudioLLM/CosyVoice.git) and [model card](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

<img width="1560" height="651" alt="image" src="https://github.com/user-attachments/assets/0dfdec7b-4d46-4a72-978c-020bc6a34764" />

### Notebook Contents

In this tutorial we consider how to run and optimize Fun-CosyVoice 3.0 using OpenVINO.

The tutorial consists of the following steps:

- Install prerequisites
- Convert model to OpenVINO intermediate representation (IR) format 
- Prepare OpenVINO Inference pipeline
- Run Speech Generation
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/cosyvoice3-tts/README.md" />
