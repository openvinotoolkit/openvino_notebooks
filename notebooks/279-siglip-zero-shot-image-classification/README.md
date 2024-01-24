# Zero-shot Image Classification with SigLIP

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/279-siglip-zero-shot-image-classification/279-siglip-zero-shot-image-classification.ipynb)

Zero-shot image classification is a computer vision task to classify images into one of several classes, without any prior training or knowledge of the classes.

![zero-shot-pipeline](https://user-images.githubusercontent.com/29454499/207773481-d77cacf8-6cdc-4765-a31b-a1669476d620.png)

In this tutorial, you will use [SigLIP](https://huggingface.co/docs/transformers/main/en/model_doc/siglip) model to perform zero-shot image classification.

## Notebook Contents

This tutorial demonstrates how to perform zero-shot image classification using the open-source SigLIP model. The SigLIP model was proposed in [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) by Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. SigLIP proposes to replace the loss function used in [CLIP](https://github.com/openai/CLIP) (Contrastive Language–Image Pre-training) by a simple pairwise sigmoid loss. This results in better performance in terms of zero-shot classification accuracy on ImageNet.

![siglip-performance-comparison](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/siglip_table.jpeg)

[\*_image source_](https://arxiv.org/abs/2303.15343)

You can find more information about this model in the [research paper](https://arxiv.org/abs/2303.15343), [GitHub repository](https://github.com/google-research/big_vision), [Hugging Face model page](https://huggingface.co/docs/transformers/main/en/model_doc/siglip).

Notebook contains the following steps:

1. Instantiate model
1. Run PyTorch model inference
1. Convert model to OpenVINO Intermediate Representation (IR) format.
1. Run OpenVINO model
1. Apply post-training quantization using NNCF
    1. Prepare dataset
    1. Quantize model
    1. Run quantized OpenVINO model
    1. Compare File Size
    1. Compare inference time of the FP16 IR and quantized models

NNCF performs quantization within the OpenVINO IR. It is required to run the first notebook before running the second notebook.

We will use SigLIP model for zero-shot image classification. The result of model work demonstrated on the image below
![image](https://user-images.githubusercontent.com/29454499/207795060-437b42f9-e801-4332-a91f-cc26471e5ba2.png)

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
