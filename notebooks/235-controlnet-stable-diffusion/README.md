# Text-to-Image Generation with ControlNet Conditioning

Diffusion models make a revolution in AI generated art. This technology enables creation of high-quality images simply by writing a text prompt. Despite the fact that this technology gives very promising results, the diffusion process, in first order, is a process of generation of images from random noise and text conditions, which do not always clarify how the desired content should look like, which forms it should have and where it is located in relation to other objects on the image. The researches have been looking for ways to have more control over the results of the generation process. ControlNet provides a minimal interface, which enables users to customize the generation process up to a great extent.

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) paper. It provides a framework that enables support for various spatial contexts such as a depth map, a segmentation map, a scribble, key points that can serve as additional conditionings to Diffusion models such as Stable Diffusion.

This notebook explores [ControlNet](https://github.com/lllyasviel/ControlNet) in depth, especially a new technique for imparting high levels of control over the shape of synthesized images. It also demonstrates how to run it, using OpenVINO. Let us get "controlling"!
The complete pipeline of this demo is shown below.

![diagram](https://user-images.githubusercontent.com/29454499/224248986-eedf6492-dd7a-402b-b65d-36de952094ec.png)

This is a demonstration, in which you can type text-based instructions and provide an input image to the pipeline that will generate a new image that reflects the context of the input text.
Step-by-step, the diffusion process will iteratively denoise the latent image representation while being conditioned on the text embeddings provided by the text encoder and human body key points detected by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) model and processed via ControlNet.

The following image shows an example of the input image, detected body key points and the corresponding result.

![image](https://user-images.githubusercontent.com/29454499/224541412-9d13443e-0e42-43f2-8210-aa31820c5b44.png)

## Notebook Contents

This notebook demonstrates how to convert and run stable diffusion using OpenVINO.

Notebook contains the following steps:
1. Create pipeline with PyTorch models using Diffusers library.
2. Convert PyTorch models to OpenVINO IR format using model conversion API.
3. Run Stable Diffusion ControlNet pipeline with OpenVINO.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).