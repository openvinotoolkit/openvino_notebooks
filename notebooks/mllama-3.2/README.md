# Visual-language assistant with Llama-3.2-11B-Vision and OpenVINO

Llama-3.2-11B-Vision is the latest model from LLama3 model family those capabilities extended to understand images content.
More details about model can be found in [model card](), [blog]() and original [repo]().

In this tutorial we consider how to convert and optimize Llama-Vision model for creating multimodal chatbot. Additionally, we demonstrate how to apply stateful transformation on LLM part and model optimization techniques like weights compression and quantization using [NNCF](https://github.com/openvinotoolkit/nncf)

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert model
- Optimize Language model using weights compression
- Optimize Image encoder using post-training quantization
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.

The image bellow illustrates example of input prompt and model answer.
![example.png](https://github.com/user-attachments/assets/1e3fde78-bae5-4b9a-8ef3-ea1291b288cf)

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/mllama-3.2/README.md" />
