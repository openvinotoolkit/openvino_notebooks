# High-Quality Text-Free One-Shot Voice Conversion with FreeVC and OpenVINO™
[FreeVC](https://github.com/OlaWod/FreeVC) allows alter the voice of a source speaker to a target style, 
while keeping the linguistic content unchanged, without text annotation.


Figure bellow illustrates model architecture of FreeVC for inference. In this notebook we concentrate only on 
inference part. There are three main parts: Prior Encoder, Speaker Encoder and Decoder. 
The prior encoder contains a WavLM model, a bottleneck extractor and a normalizing flow. 
Detailed information is available in this [paper](https://arxiv.org/abs/2210.15418).


![Inference](https://github.com/OlaWod/FreeVC/blob/main/resources/infer.png?raw=true)

[**image_source*](https://github.com/OlaWod/FreeVC)

## Notebook Contents

FreeVC suggests only command line interface to use and only with CUDA. In this notebook it shows how to use FreeVC 
in Python and without CUDA devices. It consists of the following steps:
- Download and prepare models.
- Inference.
- Convert models to OpenVINO Intermediate Representation.
- Inference using only OpenVINO's IR models.


## Installation Instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/freevc-voice-conversion/README.md" />
