# Generate creative QR codes with ControlNet QR Code Monster and OpenVINO™

[Stable Diffusion](https://github.com/CompVis/stable-diffusion), a cutting-edge image generation technique, but it can be further enhanced by combining it with [ControlNet](https://github.com/lllyasviel/ControlNet), a widely used control network approach. The combination allows Stable Diffusion to use a condition input to guide the image generation process, resulting in highly accurate and visually appealing images. The condition input could be in the form of various types of data such as scribbles, edge maps, pose key points, depth maps, segmentation maps, normal maps, or any other relevant information that helps to guide the content of the generated image, for example - QR codes! This method can be particularly useful in complex image generation scenarios where precise control and fine-tuning are required to achieve the desired results.

In this tutorial, we will learn how to convert and run [Controlnet QR Code Monster For SD-1.5](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster) by [monster-labs](https://qrcodemonster.art/). An additional part demonstrates how to run quantization with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up pipeline.

![](https://github.com/openvinotoolkit/openvino_notebooks/assets/76463150/1a5978c6-e7a0-4824-9318-a3d8f4912c47)

## Notebook Contents

This notebook demonstrates how to convert, run and optimize ControlNet and Stable Diffusion using OpenVINO and NNCF.

Notebook contains the following steps:
1. Create pipeline with PyTorch models using Diffusers library.
2. Convert PyTorch models to OpenVINO IR format using model conversion API.
3. Optimize `OVContrlNetStableDiffusionPipeline` with [NNCF](https://github.com/openvinotoolkit/nncf/) quantization.
4. Compare results of original and optimized pipelines.
5. Run Stable Diffusion ControlNet pipeline with OpenVINO.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
