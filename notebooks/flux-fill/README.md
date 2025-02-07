# Image inpainting and outpainting with FLUX.1 Fill

inpainting is the task of replacing or editing a specific area of the input image. This makes it a useful tool for image restoration like removing defects and artifacts, or even replacing an image area with something entirely new. Inpainting relies on a mask to determine which regions of an image to fill in; the area to inpaint is represented by white pixels and the area to keep is represented by black pixels. The white pixels are filled in by the prompt.
FLUX.1 Fill introduces advanced inpainting capabilities that surpass existing approaches.  It allows for seamless edits that integrate naturally with existing images. 

![](https://github.com/user-attachments/assets/3598a8e1-526b-4571-8d73-200dcde92430)

Additionally, FLUX.1 Fill supports outpainting, enabling the user to extend images beyond their original borders.

![](https://github.com/user-attachments/assets/0e195ef2-fc5d-4eca-b32f-08cdd646199f)

You can find more details about the model in [blog post](https://blackforestlabs.ai/flux-1-tools/) and [model card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)

In this tutorial, we consider how to convert and optimize FLUX.1 Fill for performing image editing using text prompt and binary mask.

### Notebook Contents

In this demonstration, you will learn how to perform inpainting and outpainting using Flux.1 and OpenVINO. 

Example of model work:

![](https://github.com/user-attachments/assets/2cc19c7c-2d68-4a33-b143-226319888bd6)

The tutorial consists of the following steps:

- Install prerequisites
- Collect Pytorch model pipeline
- Convert model to OpenVINO intermediate representation (IR) format 
- Compress weights using NNCF
- Prepare OpenVINO Inference pipeline
- Run model
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/flux-fill/README.md" />
