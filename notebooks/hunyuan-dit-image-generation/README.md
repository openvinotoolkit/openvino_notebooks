# Image generation with HunyuanDIT and OpenVINO

Hunyuan-DiT is a powerful text-to-image diffusion transformer with fine-grained understanding of both English and Chinese. The model architecture expertly blends diffusion models and transformer networks to unlock the potential of Chinese text-to-image generation.

![](https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/framework.png)

More details about model can be found in original [repository](https://github.com/Tencent/HunyuanDiT), [project web page](https://dit.hunyuan.tencent.com/) and [paper](https://arxiv.org/abs/2405.08748).

In this tutorial we consider how to convert and run Hunyuan-DIT model using OpenVINO. Additionally, we will use [NNCF](https://github.com/openvinotoolkit/nncf) for optimizing model in low precision.

The notebook provides a simple interface that allows communication with a model using text instruction on English or Chinese. In this demonstration user can provide input instructions and the model generates an image. 
The image below illustrates the provided generated image example.
![](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/b541d7d9-da82-4fe9-a98b-e744cb25c3c6)

>**Note**: Demonstrated models can require at least 32GB RAM for conversion and running.


### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Prepare Diffusers pipeline
- Convert PyTorch models and compress model weights.
- Prepare OpenVINO inference pipeline
- Run model inference
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/hunyuan-dit-image-generation/README.md" />
