# Paint by Example using Stable Diffusion and OpenVINO™

This AI painting notebook provides an interactive tool for users to create photo-realistic images based on an exemplar image and a sketch using OpenVINO. This notebook heavily leverages the "Paint by example" exemplar-based model, which is a deep learning model that has shown promising results for creating high-quality images [Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example). This is the original [white paper](https://arxiv.org/abs/2211.13227). This notebook explores how deep learning can be used to create an interactive python notebook for image semantic translation. The notebook utilizes the Paint by Example model from [HuggingFace Paint-by-Example](https://huggingface.co/Fantasy-Studio/Paint-by-Example) and performs a conversion using the ONNX export function which will produce an OpenVINO IR.


In previous notebooks, we already discussed how to run [Text-to-Image generation and Image-to-Image generation using Stable Diffusion v1](../225-stable-diffusion-text-to-image/225-stable-diffusion-text-to-image.ipynb) and [controlling its generation process using ControlNet](../235-controlnet-stable-diffusion/235-controlnet-stable-diffusion.ipynb).

This notebook demonstrates in-painting using a reference image as input and a hand-drawn mask:

This is a demonstration in which you can select a source image and then use a drawing tool to create a mask of the area you want the model to draw inside. Next select a reference image to feed into the model and the pipeline will generate an image inside the masked region that reflects the context of the reference image.
Step-by-step, the diffusion process will iteratively denoise latent image representation while being conditioned on the context of the reference image provided to the image encoder.

The following image shows an example of the source image with a mask drawn over the top of it. Below it is the reference image used for input context. Lastly, is shown the output image.

![Diagram](https://user-images.githubusercontent.com/103226580/235281192-66eeefee-6c6a-45af-b805-0eb10490f78e.png)

## Notebook Contents

This notebook demonstrates how to convert and run the Paint-by-Example model using OpenVINO.

Notebook contains the following steps:
1. Convert PyTorch models to OpenVINO IR format.
3. Run Paint-by-Example pipeline with OpenVINO.

## Installation instructions
If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
