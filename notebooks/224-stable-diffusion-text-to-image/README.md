# Text-to-image generation with Stable Diffusion and OpenVINO

Text-to-image generation is task of AI, which aim is create image from given text description.
In this demo we use the **[Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion)** model for image generation with OpenVINO.

The complete pipeline of this demo's notebook is shown below.

![image2](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/stable_diffusion.png)

This is a demonstration in which the user can type the text and the pipeline will generate image which correspond to given text, step by step iteratively denoise latent image representation while being conditioned on the text embeddings provided by text encoder. 

The following image show an example of the input sequence and corresponding predicted image.

![image](data/Corgi.png)

## Notebook Contents

This notebook demostrates how convert and run stable diffusion using OpenVINO.

Notebook contains following steps:
1. Convert model to ONNX format
2. Convert model to IR using OpenVINO Model Optimizer
3. Run Stable Diffusion pipeline with OpenVINO models

## Model Access

You need to accept the model license before downloading or using the weights. In this demo we'll use model version `v1-4`, so you'll need to  visit [its card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree. 

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md") to install all required dependencies.