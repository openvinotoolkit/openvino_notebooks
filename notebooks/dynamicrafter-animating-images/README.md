# Animating Open-domain Images with DynamiCrafter and OpenVINO

Animating a still image offers an engaging visual experience. Traditional image animation techniques mainly focus on animating natural scenes with stochastic dynamics (e.g. clouds and fluid) or domain-specific motions (e.g. human hair or body motions), and thus limits their applicability to more general visual content. To overcome this limitation, [DynamiCrafter team](https://doubiiu.github.io/projects/DynamiCrafter/) explores the synthesis of dynamic content for open-domain images, converting them into animated videos. The key idea is to utilize the motion prior of text-to-video diffusion models by incorporating the image into the generative process as guidance. Given an image, DynamiCrafter team first projects it into a text-aligned rich context representation space using a query transformer, which facilitates the video model to digest the image content in a compatible fashion. However, some visual details still struggle to be preserved in the resultant videos. To supplement with more precise image information, DynamiCrafter team further feeds the full image to the diffusion model by concatenating it with the initial noises. Experimental results show that the proposed method can produce visually convincing and more logical & natural motions, as well as higher conformity to the input image.

<table class="center">
  <tr>
    <td colspan="2">"bear playing guitar happily, snowing"</td>
    <td colspan="2">"boy walking on the street"</td>
  </tr>
  <tr>
  <td>
    <img src=https://github.com/Doubiiu/DynamiCrafter/blob/main/assets/showcase/guitar0.jpeg_00.png?raw=True width="170">
  </td>
  <td>
    <img src=https://github.com/Doubiiu/DynamiCrafter/blob/main/assets/showcase/guitar0.gif?raw=True width="170">
  </td>
  <td>
    <img src=https://github.com/Doubiiu/DynamiCrafter/blob/main/assets/showcase/walk0.png_00.png?raw=True width="170">
  </td>
  <td>
    <img src=https://github.com/Doubiiu/DynamiCrafter/blob/main/assets/showcase/walk0.gif?raw=True width="170">
  </td>
  </tr>
</table >

## Notebook contents
This tutorial consists of the following steps:
- Prerequisites
- Load the original model
- Convert the model to OpenVINO IR
- Compiling models
- Building the pipeline
- Optimize pipeline with [NNCF](https://github.com/openvinotoolkit/nncf/)
- Compare results of original and optimized pipelines
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/dynamicrafter-animating-images/README.md" />
