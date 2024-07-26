# Sound Generation with Stable Audio Open and OpenVINO™

[Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) is an open-source model optimized for generating short audio samples, sound effects, and production elements using text prompts. The model was trained on data from Freesound and the Free Music Archive, respecting creator rights.

<img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/76171391/ed4aa0f2-0501-4519-8b24-c1c3072b4ef2" />

#### Key Takeaways:

 - Stable Audio Open is an open source text-to-audio model for generating up to 47 seconds of samples and sound effects.
 - Users can create drum beats, instrument riffs, ambient sounds, foley and production elements.
 - The model enables audio variations and style transfer of audio samples.

This model is made to be used with the [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) library for inference.

## Notebook contents
This tutorial consists of the following steps:
- Prerequisites
- Load the original model and inference
- Convert the model to OpenVINO IR
- Compiling models and inference
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/stable-audio/README.md" />
