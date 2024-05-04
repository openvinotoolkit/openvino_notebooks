# Zero-shot Image Classification with SigLIP

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/282-siglip-zero-shot-image-classification/282-siglip-zero-shot-image-classification.ipynb)

Zero-shot image classification is a computer vision task with the goal to classify images into one of several classes without any prior training or knowledge of these classes.

![zero-shot-pipeline](https://user-images.githubusercontent.com/29454499/207773481-d77cacf8-6cdc-4765-a31b-a1669476d620.png)

In this tutorial, you will use the [SigLIP](https://huggingface.co/docs/transformers/main/en/model_doc/siglip) model to perform zero-shot image classification.

## Notebook Contents

This tutorial demonstrates how to perform zero-shot image classification using the open-source SigLIP model. The SigLIP model was proposed in the [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) paper. SigLIP suggests replacing the loss function used in [CLIP](https://github.com/openai/CLIP) (Contrastive Language–Image Pre-training) with a simple pairwise sigmoid loss. This results in better performance in terms of zero-shot classification accuracy on ImageNet.

![siglip-performance-comparison](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/siglip_table.jpeg)

[\*_image source_](https://arxiv.org/abs/2303.15343)

You can find more information about this model in the [research paper](https://arxiv.org/abs/2303.15343), [GitHub repository](https://github.com/google-research/big_vision), [Hugging Face model page](https://huggingface.co/docs/transformers/main/en/model_doc/siglip).

The notebook contains the following steps:

1. Instantiate model.
1. Run PyTorch model inference.
1. Convert the model to OpenVINO Intermediate Representation (IR) format.
1. Run OpenVINO model.
1. Apply post-training quantization using [NNCF](https://github.com/openvinotoolkit/nncf):
   1. Prepare dataset.
   1. Quantize model.
   1. Run quantized OpenVINO model.
   1. Compare File Size.
   1. Compare inference time of the FP16 IR and quantized models.

The results of the SigLIP model's performance in zero-shot image classification from this notebook are demonstrated in the image below.
![image](https://github.com/openvinotoolkit/openvino_notebooks/assets/67365453/c4eb782c-0fef-4a89-a5c6-5cc43518490b)

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
