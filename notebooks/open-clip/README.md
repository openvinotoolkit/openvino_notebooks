# Running OpenCLIP models using OpenVINO™

[OpenCLIP](https://github.com/mlfoundations/open_clip) is an open source implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pre-training).

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on various (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task.
CLIP uses a [ViT](https://arxiv.org/abs/2010.11929) like transformer to get visual features and a causal language model to get the text features. The text and visual features are then projected into a latent space with identical dimensions. The dot product between the projected image and text features is then used as a similarity score.

![clip](https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png)

[**image_source*](https://github.com/openai/CLIP/blob/main/README.md)

You can find more information about this model in the [research paper](https://arxiv.org/abs/2103.00020) and GitHub [repository](https://github.com/mlfoundations/open_clip).


OpenCLIP could be used for calculation the similarity between arbitrary image, text inputs and perform zero-shot image classifications. 
In this tutorial, we will use the [OpenCLIP](https://github.com/mlfoundations/open_clip) model to try out the different tasks it is intended for. Additionally, the notebook demonstrates how to optimize the model using [NNCF](https://github.com/openvinotoolkit/nncf/).

The notebook contains the following steps:

1. Download the model.
2. Instantiate the PyTorch model.
3. Convert model to OpenVINO IR, using the model conversion API.
4. Run inference with OpenVINO.
5. Quantize the converted model with NNCF.
6. Check the quantized model inference result.
7. Compare model size of converted and quantized models.
8. Compare performance of converted and quantized models.
9. Launch interactive demo


We will try OpenCLIP model for different task, but we will pay the most attention to zero-shot image classification. The example of result of the model work are demonstrated on the image below
![image](https://user-images.githubusercontent.com/29454499/207795060-437b42f9-e801-4332-a91f-cc26471e5ba2.png)

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md)..

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/open-clip/README.md" />
