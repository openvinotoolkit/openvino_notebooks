# Mobile language assistant with MobileVLM and OpenVINO

[MobileVLM](https://arxiv.org/abs/2312.16886) is a competent multimodal vision language model (MMVLM) targeted to run on mobile devices. It is an amalgamation of a myriad of architectural designs and techniques that are mobile-oriented, which comprises a set of language models at the scale of 1.4B and 2.7B parameters, trained from scratch, a multimodal vision model that is pre-trained in the CLIP fashion, cross-modality interaction via an efficient projector.

![](https://github.com/Meituan-AutoML/MobileVLM/raw/main/assets/mobilevlm_arch.png)

The MobileVLM architecture (right) utilizes MobileLLaMA as its language model, intakes $\mathbf{X}_v$ and $\mathbf{X}_q$ which are image and language instructions as respective inputs and gives $\mathbf{Y}_a$ as the output language response. LDP refers to a lightweight downsample projector (left).

See more information on official [GitHub](https://github.com/Meituan-AutoML/MobileVLM) project page.

In this tutorial we consider how to use MobileVLM model to build multimodal language assistant with OpenVINO help.

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Clone MobileVLM repository
- Import required packages
- Load the model
- Convert model to OpenVINO Intermediate Representation (IR)
- Inference
    - Load OpenVINO model
    - Prepare input data
    - Run generation process
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).