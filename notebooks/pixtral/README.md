 Visual-language assistant with Pixtral and OpenVINO

Pixtral-12b is multimodal model that consists of 12B parameter multimodal decoder based on Mistral Nemo and 400M parameter vision encoder trained from scratch. It is trained to understand both natural images and documents. The model shows strong abilities in tasks such as chart and figure understanding, document question answering, multimodal reasoning and instruction following. Pixtral is able to ingest images at their natural resolution and aspect ratio, giving the user flexibility on the number of tokens used to process an image. Pixtral is also able to process any number of images in its long context window of 128K tokens. Unlike previous open-source models, Pixtral does not compromise on text benchmark performance to excel in multimodal tasks.

![](https://mistral.ai/images/news/pixtral-12b/pixtral-model-architecture.png)


More details about model are available in [blog post](https://mistral.ai/news/pixtral-12b/) and [model card](https://huggingface.co/mistralai/Pixtral-12B-2409)

In this tutorial we consider how to convert, optimize and run this model using OpenVINO.

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.

The image bellow illustrates example of input prompt and model answer.
![example.png](https://github.com/user-attachments/assets/b61a9e8e-32c7-4b60-aa00-b4b867e823be)

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/pixtral/README.md" />
