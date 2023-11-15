# Image generation with DeepFloyd IF and OpenVINO™

DeepFloyd IF is an advanced open-source text-to-image model that delivers remarkable photorealism and language comprehension. DeepFloyd IF consists of a frozen text encoder and three cascaded pixel diffusion modules: a base model that creates 64x64 px images based on text prompts and two super-resolution models, each designed to generate images with increasing resolution: 256x256 px and 1024x1024 px. All stages of the model employ a frozen text encoder, built on the T5 transformer, to derive text embeddings, which are then passed to a UNet architecture enhanced with cross-attention and attention pooling.

![deepfloyd_if_scheme](https://github.com/deep-floyd/IF/raw/develop/pics/deepfloyd_if_scheme.jpg)

## Notebook Contents

This folder contains two notebooks that show how to convert, run and optimize models using OpenVINO.

The [first notebook](238-deep-floyd-if-convert.ipynb) is about the conversion to IR and consists of following steps:
1. Convert PyTorch models to OpenVINO IR format, using model conversion API.
2. Run DeepFloyd IF pipeline with OpenVINO.

The result of notebook work demonstrated on the image below:
![owl.png](https://user-images.githubusercontent.com/29454499/241643886-dfcf3c48-8d50-4730-ae28-a21595d9504f.png)

>**Note**: Please be aware that a machine with at least 32GB of RAM is necessary to run this example.

The [second notebook](238-deep-floyd-if-optimize.ipynb) is about the optimization by 8-bit post-training quantization and weights compression and consists of the following steps:
1. Compress weights of the converted OpenVINO text encoder from the [first notebook](238-deep-floyd-if-convert.ipynb) with NNCF.
2. Quantize the converted stage_1 and stage_2 U-Nets from the [first notebook](238-deep-floyd-if-convert.ipynb) with NNCF.
2. Check the model result using text prompts from the [first notebook](238-deep-floyd-if-convert.ipynb) .
3. Compare model size of converted and optimized models.
4. Compare performance of converted and optimized models.

>**Note**: NNCF performs optimizations within the OpenVINO IR. It is required to run the first notebook before running the second notebook.

# Installation Instructions

The Jupyter notebook contains its own set of requirements installed directly within the notebook, allowing it to run independently as a standalone example.