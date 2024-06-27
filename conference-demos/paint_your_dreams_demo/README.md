# Image generation with Latent Consistency Model and OpenVINO

LCMs: The next generation of generative models after Latent Diffusion Models (LDMs). 
Latent Diffusion models (LDMs) have achieved remarkable results in synthesizing high-resolution images. However, the iterative sampling is computationally intensive and leads to slow generation.

**Input text:** a beautiful pink unicorn, 8k

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/277367065-13a8f622-8ea7-4d12-b3f8-241d4499305e.png"/>
</p>

### Notebook Contents

This [notebook](./263-lcm-image-generation-paint-your-dreams.ipynb) demonstrates how to convert and run [LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) using OpenVINO. An additional part demonstrates how to run quantization with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up pipeline. Also it compares those models runned on CPU and GPU.

The notebook contains the following steps:

1. Convert PyTorch models to OpenVINO Intermediate Representation using [OpenVINO Model Conversion API](https://docs.openvino.ai/2023.2/openvino_docs_model_processing_introduction.html#convert-a-model-with-python-convert-model)
2. Prepare Inference Pipeline.
3. Run Inference pipeline with OpenVINO.
4. Optimize `LatentConsistencyModelPipeline` with [NNCF](https://github.com/openvinotoolkit/nncf/) quantization.
5. Compare results of original, optimized and GPU pipelines.
6. Run Interactive demo.

The notebook also provides interactive interface for image generation based on user input.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

## Docker Installation Instructions

1. Open terminal in "openvino_notebooks" root directory
2. Build docker image: `sudo docker build -f conference-demos/paint-your-dreams-demo/Dockerfile -t your_image_name .`
3. Run docker
    1. CPU ONLY: `sudo docker run -it -p 8888:8888 -p 7860-7870:7860-7870 your_image_name`
    2. CPU+GPU: `sudo docker run -it --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -p 8888:8888 -p 7860-7870:7860-7870 your_image_name`

*8888 port opens port for jupyter*

*7860 is default port for Gradio Apps*



