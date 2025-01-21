# Image-to-image generation using OpenVINO GenAI

Image-to-image is the task of transforming an input image through a variety of possible manipulations and enhancements, such as super-resolution, image inpainting, colorization, stylization and more.

One of the most popular use cases of image-to-image is style transfer. With style transfer models:
  * a regular photo can be transformed into a variety of artistic styles or genres, such as a watercolor painting, a comic book illustration and more.
  * new images can be generated using a text prompt, in the style of a reference input image.
  
Latent diffusion models can be used for performing image-to-image generation. Diffusion-based Image-to-image is similar to [text-to-image](../text-to-image-genai/text-to-image-genai.ipynb), but in addition to a prompt, you can also pass an initial image as a starting point for the diffusion process. The initial image is encoded to latent space and noise is added to it. Then the latent diffusion model takes a prompt and the noisy latent image, predicts the added noise, and removes the predicted noise from the initial latent image to get the new latent image. Lastly, a decoder decodes the new latent image back into an image.

![pipe.png](https://user-images.githubusercontent.com/29454499/260981188-c112dd0a-5752-4515-adca-8b09bea5d14a.png)

In this tutorial, we consider how to use OpenVINO GenAI for performing image-to-image generation.

## About OpenVINO GenAI

[OpenVINO™ GenAI](https://github.com/openvinotoolkit/openvino.genai) is a library of the most popular Generative AI model pipelines, optimized execution methods, and samples that run on top of highly performant OpenVINO Runtime.

This library is friendly to PC and laptop execution, and optimized for resource consumption. It requires no external dependencies to run generative models as it already includes all the core functionality (e.g. tokenization via openvino-tokenizers).

OpenVINO GenAI supports popular diffusion models like Stable Diffusion or SDXL for performing image generation. You can find supported models list in [OpenVINO GenAI documentation](https://github.com/openvinotoolkit/openvino.genai/blob/master/SUPPORTED_MODELS.md#image-generation-models). Previously, we considered how to run text-to-image generation with OpenVINO GenAI and apply multiple LoRA adapters, mow is image-to-image. 

## Notebook Contents

In this notebook we will demonstrate how to use Latent Diffusion models like Stable Diffusion 1.5, 2.1, LCM, SDXL for image to image generation using OpenVINO GenAI Image2ImagePipeline. 
All it takes is two steps: 
1. Export OpenVINO IR format model using the [Hugging Face Optimum](https://huggingface.co/docs/optimum/installation) library accelerated by OpenVINO integration.
The Hugging Face Optimum Intel API is a high-level API that enables us to convert and quantize models from the Hugging Face Transformers library to the OpenVINO™ IR format. For more details, refer to the [Hugging Face Optimum Intel documentation](https://huggingface.co/docs/optimum/intel/inference).
1. Run inference using the standard [Image-to-Image Generation pipeline](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html) from OpenVINO GenAI.

The tutorial consists of following steps:
- Install prerequisites
- Prepare models
  - Convert models using HuggingFace Optimum Intel
  - Obtain optimized models from HuggingFace Hub
- Prepare Inference pipeline
- Run image-to-image generation
- Explore advanced options for generation results improvement
- Launch interactive demo

![](https://github.com/user-attachments/assets/280736ea-d51a-43f3-a1ae-21af5831005f) 


## Installation Instructions
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/image-to-image-genai/README.md" />
