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
If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

Also, it requires some extra steps, that can be done manually or will be performed automatically during the execution 
of the notebook, but in minimum necessary scope. 
1. Clone this repo: `git clone https://github.com/OlaWod/FreeVC.git`.
2. Download [WavLM-Large](https://github.com/microsoft/unilm/tree/master/wavlm) and put it under `FreeVC/wavlm/` directory.
3. You can download the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset. For this example we download only
two of them from [Hugging Face FreeVC example](https://huggingface.co/spaces/OlaWod/FreeVC/tree/main).
4. Download [pretrained models](https://1drv.ms/u/s!AnvukVnlQ3ZTx1rjrOZ2abCwuBAh?e=UlhRR5) and put it under `checkpoints` directory (for current example only `freevc.pth` are required).