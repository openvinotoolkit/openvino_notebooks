# Zero-shot Image Classification with OpenAI CLIP
Zero-shot image classification is a computer vision task to classify images into one of several classes, without any prior training or knowledge of the classes.

![zero-shot-pipeline](https://user-images.githubusercontent.com/29454499/207773481-d77cacf8-6cdc-4765-a31b-a1669476d620.png)

In this tutorial, you will use [OpenAI CLIP](https://github.com/openai/CLIP) model to perform zero-shot image classification.

## Notebook Contents

This tutorial demonstrates how to perform zero-shot image classification using the open-source CLIP model. CLIP is a multi-modal vision and language model. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task. According to the [paper](https://arxiv.org/abs/2103.00020), CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.
You can find more information about this model in the [research paper](https://arxiv.org/abs/2103.00020), [OpenAI blog](https://openai.com/blog/clip/), [model card](https://github.com/openai/CLIP/blob/main/model-card.md) and GitHub [repository](https://github.com/openai/CLIP).

This folder contains two notebooks that show how to convert and quantize model with OpenVINO:

1. [Convert the CLIP model using OpenVINO](228-clip-zero-shot-convert.ipynb)
2. [Quantize the OpenVINO CLIP model using NNCF](228-clip-zero-shot-quantize.ipynb)

The first notebook contains the following steps:
1. Download the model.
2. Instantiate the PyTorch model.
3. Convert model to OpenVINO IR, using the model conversion API.
4. Run CLIP with OpenVINO.

The second notebook contains the following steps:
1. Quantize the converted OpenVINO model from [notebook](228-clip-zero-shot-convert.ipynb) with NNCF.
2. Check the model result using the same input data from the [first notebook](228-clip-zero-shot-convert.ipynb).
3. Compare model size of converted and quantized models.
4. Compare performance of converted and quantized models.

NNCF performs quantization within the OpenVINO IR. It is required to run the first notebook before running the second notebook.

We will use CLIP model for zero-shot image classification. The result of model work demonstrated on the image below
![image](https://user-images.githubusercontent.com/29454499/207795060-437b42f9-e801-4332-a91f-cc26471e5ba2.png)

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
