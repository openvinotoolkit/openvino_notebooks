# Zero-shot Image Classification with SigLIP2

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/siglip-zero-shot-image-classification/siglip-zero-shot-image-classification.ipynb)

Zero-shot image classification is a computer vision task with the goal to classify images into one of several classes without any prior training or knowledge of these classes.

![zero-shot-pipeline](https://user-images.githubusercontent.com/29454499/207773481-d77cacf8-6cdc-4765-a31b-a1669476d620.png)

In this tutorial, you will use the [SigLIP2](https://huggingface.co/blog/siglip2) model to perform zero-shot image classification.

## Notebook Contents

The SigLIP model was proposed in [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343). SigLIP proposes to replace the loss function used in [CLIP](https://github.com/openai/CLIP) (Contrastive Language–Image Pre-training) by a simple pairwise sigmoid loss. This results in better performance in terms of zero-shot classification accuracy on ImageNet.

The abstract from the paper is the following:

> We propose a simple pairwise Sigmoid loss for Language-Image Pre-training (SigLIP). Unlike standard contrastive learning with softmax normalization, the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization. The sigmoid loss simultaneously allows further scaling up the batch size, while also performing better at smaller batch sizes.

You can find more information about this model in the [research paper](https://arxiv.org/abs/2303.15343), [GitHub repository](https://github.com/google-research/big_vision), [Hugging Face model page](https://huggingface.co/docs/transformers/main/en/model_doc/siglip).

[SigLIP 2](https://huggingface.co/papers/2502.14786) extends the pretraining objective of SigLIP with prior, independently developed techniques into a unified recipe, for improved semantic understanding, localization, and dense features. SigLIP 2 models outperform the older SigLIP ones at all model scales in core capabilities, including zero-shot classification, image-text retrieval, and transfer performance when extracting visual representations for Vision-Language Models (VLMs). More details about SigLIP 2 can be found in [blog post](https://huggingface.co/blog/siglip2)

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/sg2-blog/decoder.png).

SigLIP 2 models outperform the older SigLIP ones at all model scales in core capabilities, including zero-shot classification, image-text retrieval, and transfer performance when extracting visual representations for Vision-Language Models (VLMs).
A cherry on top is the dynamic resolution (naflex) variant. This is useful for downstream tasks sensitive to aspect ratio and resolution.

In this notebook, we will use [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) by default, but the same steps are applicable for other SigLIP family models.

The notebook contains the following steps:

1. Instantiate model.
2. Run PyTorch model inference.
3. Convert the model to OpenVINO Intermediate Representation (IR) format.
4. Run OpenVINO model.
5. Apply post-training quantization using [NNCF](https://github.com/openvinotoolkit/nncf):
   1. Prepare dataset.
   2. Quantize model.
   3. Run quantized OpenVINO model.
   4. Compare File Size.
   5. Compare inference time of the FP16 IR and quantized models.

The results of the SigLIP model's performance in zero-shot image classification from this notebook are demonstrated in the image below.
![image](https://github.com/openvinotoolkit/openvino_notebooks/assets/67365453/c4eb782c-0fef-4a89-a5c6-5cc43518490b)

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/siglip-zero-shot-image-classification/README.md" />
