# InstantID: Zero-shot Identity-Preserving Generation using OpenVINO

Nowadays has been significant progress in personalized image synthesis with methods such as Textual Inversion, DreamBooth, and LoRA.
However, their real-world applicability is hindered by high storage demands, lengthy fine-tuning processes, and the need for multiple reference images. Conversely, existing ID embedding-based methods, while requiring only a single forward inference, face challenges: they either necessitate extensive fine-tuning across numerous model parameters, lack compatibility with community pre-trained models, or fail to maintain high face fidelity. 

[InstantID](https://instantid.github.io/) is a tuning-free method to achieve ID-Preserving generation with only single image, supporting various downstream tasks.
![applications.png](https://github.com/InstantID/InstantID/blob/main/assets/applications.png?raw=true)

Given only one reference ID image, InstantID aims to generate customized images with various poses or styles from a single reference ID image while ensuring high fidelity. Following figure provides an overview of the method. It incorporates three crucial components: 

1. An ID embedding that captures robust semantic face information; 
2. A lightweight adapted module with decoupled cross-attention, facilitating the use of an image as a visual prompt;
3. An IdentityNet that encodes the detailed features from the reference facial image with additional spatial control.

![instantid-components.png](https://instantid.github.io/static/documents/pipeline.png)

The difference InstantID from previous works in the following aspects: 
1. do not involve UNet training, so it can preserve the generation ability of the original text-to-image model and be compatible with existing pre-trained models and ControlNets in the community;
2. doesn't require test-time tuning, so for a specific character, there is no need to collect multiple images for fine-tuning, only a single image needs to be inferred once;
3. achieve better face fidelity, and retain the editability of text.

You can find more details about the approach with [project web page](https://instantid.github.io/), [paper](https://arxiv.org/abs/2401.07519) and [original repository](https://github.com/InstantID/InstantID)

In this tutorial, we consider how to use InstantID with OpenVINO.

We will use a pre-trained model from the [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) library.

The notebook provides a simple interface that allows communication with a model using text instruction and images. In this demonstration user can provide input instructions and image and the model generates an image. An additional part demonstrates how to run optimization with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up pipeline.
The image below illustrates the provided generated image example.

![generation_example.png](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/082b3da7-6bb6-4551-bfa6-0e43d8e80b51)

>**Note**: Some demonstrated models can require at least 32GB RAM for conversion and running.

### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Prepare Face analysis pipeline with OpenVINO
- Prepare Diffusers pipeline
- Convert PyTorch models
- Prepare OpenVINO inference pipeline
- Run model inference
- Optimize pipeline with [NNCF](https://github.com/openvinotoolkit/nncf/)
- Compare results of original and optimized pipelines
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
