# Image Generation with Stable Diffusion

This notebook demonstrates how to use a **[Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion)** model for image generation with OpenVINO.
It considers two approaches of image generation using an AI method called `diffusion`:

* `Text-to-image` generation to create images from a text description as input.
* `Text-guided Image-to-Image` generation to create an image, using text description and initial image semantic.

The complete pipeline of this demo is shown below.

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/260981188-c112dd0a-5752-4515-adca-8b09bea5d14a.png"/>
</p>


This is a demonstration in which you can type a text description (and provide input image in case of Image-to-Image generation) and the pipeline will generate an image that reflects the context of the input text.
Step-by-step, the diffusion process will iteratively denoise latent image representation while being conditioned on the text embeddings provided by the text encoder.

The following image shows an example of the input sequence and corresponding predicted image.

**Input text:** cyberpunk cityscape like Tokyo, New York with tall buildings at dusk golden hour cinematic lighting, epic composition. A golden daylight, hyper-realistic environment. Hyper and intricate detail, photo-realistic. Cinematic and volumetric light. Epic concept art. Octane render and Unreal Engine, trending on artstation

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/216524089-ed671fc7-a78b-42bf-aa96-9f7c791a9419.png"/>
</p>

## Notebook Contents

This notebook demonstrates how to convert and run stable diffusion using OpenVINO.  For user experience simplification, we will use  [Hugging Face Optimum](https://huggingface.co/docs/optimum/installation) library accelerated by OpenVINO integration for model conversion and [OpenVINO GenAI API](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html) for model inference.

Notebook contains the following parts:
1. Download the model from the Hugging Face Hub and converted to OpenVINO IR format with [Optimum Intel](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion).
2. Prepare text-to-image inference pipeline and demonstrate generation with a demo.
3. Prepare image-to-image inference pipeline and demonstrate generation with a demo.


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/stable-diffusion-text-to-image/README.md" />
