# Visual-language assistant with Gemma3 and OpenVINO GenAI
![](https://github.com/user-attachments/assets/2540a58e-c242-4439-b151-0fd1e6938af1)

Gemma 3 is Google's new iteration of open weight LLMs. It comes in four sizes, 1 billion, 4 billion, 12 billion, and 27 billion parameters with base (pre-trained) and instruction-tuned versions. The 4, 12, and 27 billion parameter models can process both images and text, while the 1B variant is text only.

The input context window length has been increased from Gemma 2’s 8k to 32k for the 1B variants, and 128k for all others. As is the case with other VLMs (vision-language models), Gemma 3 generates text in response to the user inputs, which may consist of text and, optionally, images. Example uses include question answering, analyzing image content, summarizing documents, etc.

The three core enhancements in Gemma 3 over Gemma 2 are:
* Longer context length
* Multimodality
* Multilinguality

You can find more details about model in the [blog post](https://developers.googleblog.com/en/introducing-gemma3/).

In this tutorial we consider how to convert and optimize Gemma3 model for creating multimodal chatbot using [Optimum Intel](https://github.com/huggingface/optimum-intel). Additionally, we demonstrate how to apply model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf).

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO GenAI model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.
The image bellow illustrates example of input prompt and model answer.
![example.png](https://github.com/user-attachments/assets/7d886eb4-af78-4d3a-bfc3-8ae1c7147f31)

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/gemma3/README.md" />
