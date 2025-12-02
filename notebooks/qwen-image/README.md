# Text-to-image generation with Qwen-Image and OpenVINO

Qwen-Image, an image generation foundation model in the Qwen series that achieves significant advances in complex text rendering and precise image editing. Experiments show strong general capabilities in both image generation and editing, with exceptional performance in text rendering, especially for Chinese. More details about model can be found in [blog post](https://qwenlm.github.io/blog/qwen-image/) and [model card](https://huggingface.co/Qwen/Qwen-Image).


<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>

In this tutorial we consider how to convert and optimize Qwen-Image model using OpenVINO.

>**Note**: Some demonstrated models can require at least 64GB RAM for conversion and running.

### Notebook Contents

In this demonstration, you will learn how to perform text-to-image generation using Qwen-Image and OpenVINO. 

The tutorial consists of the following steps:

- Install prerequisites
- Collect Pytorch model pipeline
- Convert model to OpenVINO intermediate representation (IR) format 
- Compress weights using NNCF
- Prepare OpenVINO Inference pipeline
- Run Text-to-Image generation
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen-image/README.md" />
