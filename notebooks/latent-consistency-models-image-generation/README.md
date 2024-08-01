# Image generation with Latent Consistency Model and OpenVINO

LCMs: The next generation of generative models after Latent Diffusion Models (LDMs).
Latent Diffusion models (LDMs) have achieved remarkable results in synthesizing high-resolution images. However, the iterative sampling is computationally intensive and leads to slow generation.

Inspired by [Consistency Models](https://arxiv.org/abs/2303.01469), [Latent Consistency Models](https://arxiv.org/pdf/2310.04378.pdf) (LCMs) were proposed, enabling swift inference with minimal steps on any pre-trained LDMs, including Stable Diffusion.
The [Consistency Model (CM) (Song et al., 2023)](https://arxiv.org/abs/2303.01469) is a new family of generative models that enables one-step or few-step generation. The core idea of the CM is to learn the function that maps any points on a trajectory of the PF-ODE (probability flow of [ordinary differential equation](https://en.wikipedia.org/wiki/Ordinary_differential_equation)) to that trajectory’s origin (i.e., the solution of the PF-ODE). By learning consistency mappings that maintain point consistency on ODE-trajectory, these models allow for single-step generation, eliminating the need for computation-intensive iterations. However, CM is constrained to pixel space image generation tasks, making it unsuitable for synthesizing high-resolution images. LCMs adopt a consistency model in the image latent space for generation high-resolution images.  Viewing the guided reverse diffusion process as solving an augmented probability flow ODE (PF-ODE), LCMs are designed to directly predict the solution of such ODE in latent space, mitigating the need for numerous iterations and allowing rapid, high-fidelity sampling. Utilizing image latent space in large-scale diffusion models like Stable Diffusion (SD) has effectively enhanced image generation quality and reduced computational load. The authors of LCMs provide a simple and efficient one-stage guided consistency distillation method named Latent Consistency Distillation (LCD) to distill SD for few-step (2∼4) or even 1-step sampling and propose the SKIPPING-STEP technique to further accelerate the convergence. More details about the proposed approach and models can be found using the following resources: [project page](https://latent-consistency-models.github.io/), [paper](https://arxiv.org/abs/2310.04378), [original repository](https://github.com/luosiallen/latent-consistency-model).

In this[tutorial, we consider how to convert and run LCM using OpenVINO. We consider 2 examples of using LCM:

## Using models prepared with Latent Consistency Distillation

This notebook is a demonstration in which you can type a text description and the pipeline will generate an image that reflects the context of the input text, using model distilled using [Latent Consistency Distillation (LCD)](https://arxiv.org/abs/2310.04378) approach.
Step-by-step, the diffusion process will iteratively denoise latent image representation while being conditioned on the text embeddings provided by the text encoder.

The following image shows an example of the input sequence and corresponding predicted image.



**Input text:** a beautiful pink unicorn, 8k

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/277367065-13a8f622-8ea7-4d12-b3f8-241d4499305e.png"/>
</p>

### Notebook Contents

This [notebook](./latent-consistency-models-image-generation.ipynb) demonstrates how to convert and run [LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) using OpenVINO. An additional part demonstrates how to run quantization with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up pipeline.

The notebook contains the following steps:

1. Convert PyTorch models to OpenVINO Intermediate Representation using [OpenVINO Model Conversion API](https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model)
2. Prepare Inference Pipeline.
3. Run Inference pipeline with OpenVINO.
4. Optimize `LatentConsistencyModelPipeline` with [NNCF](https://github.com/openvinotoolkit/nncf/) quantization.
5. Compare results of original and optimized pipelines.
6. Run Interactive demo.

The notebook also provides interactive interface for image generation based on user input

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/284298771-708cc873-ccea-40b8-a681-de1ddc97ae94.gif"/>
</p>

## Using models prepared using LCM LoRA

The LCM LoRA allows to training of just a small number of adapters, known as LoRA layers, instead of the full model. The resulting LoRAs can then be applied to any fine-tuned version of the model without having to distill them separately. The benefit of this LCM LoRA distillation process is that it can be integrated into the existing inference pipelines without changes to the main code, for example, into the ControlNet-guided Stable Diffusion pipeline. More details about LCM LoRA can be found in the [technical report](https://arxiv.org/abs/2311.05556) and [blog post](https://huggingface.co/blog/lcm_lora)


### Notebook Contents

This [notebook](./lcm-lora-controlnet.ipynb) explores how to speed up ControlNet pipeline using LCM LoRA and OpenVINO. It demonstrates how to apply [LCM LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) and convert and run [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) conditioned by [ConrolNet Normal Mapping](https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae) using OpenVINO. An additional part demonstrates how to run quantization with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up pipeline.

The following image shows an example of the input sequence, input image, condition image and corresponding predicted image.



**Input text:** A head full of roses

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/284291731-9c4ee978-a934-43ea-9484-b6b91a9f9407.png"/>
</p>

1. Convert PyTorch models to OpenVINO Intermediate Representation using [OpenVINO Model Conversion API](https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model)
2. Prepare Inference Pipeline.
3. Run Inference pipeline with OpenVINO.
4. Optimize `OVControlNetStableDiffusionPipeline` with [NNCF](https://github.com/openvinotoolkit/nncf/) quantization.
5. Compare results of original and optimized pipelines.
6. Run Interactive demo.

The notebook also provides interactive interface for image generation based on user input.

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/284299074-8ba6c6c4-d8fa-4e5a-9d5d-d76f430f5647.gif"/>
</p>

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/latent-consistency-models-image-generation/README.md" />
