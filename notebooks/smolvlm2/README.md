# Visual-Language Assistant with SmolVLM2 and OpenVINO

SmolVLM2 represents a fundamental shift in how we think about video understanding - moving from massive models that require substantial computing resources to efficient models that can run anywhere.

![](https://github.com/user-attachments/assets/0f5ab933-a0e9-4f64-a572-a230c50b5495)

Its goal is simple: make video understanding accessible across all devices and use cases, from phones to servers.

Compared with the previous SmolVLM family, SmolVLM2 2.2B model got better at solving math problems with images, reading text in photos, understanding complex diagrams, and tackling scientific visual questions.

You can find more details about model in [model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) and [HuggingFace blog post](https://huggingface.co/blog/smolvlm2)

In this tutorial we consider how to convert and optimize SmolVLM2 model for creating multimodal chatbot using [Optimum Intel](https://github.com/huggingface/optimum-intel). Additionally, we demonstrate how to apply model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf).

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's or video's content.

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/smolvlm2/README.md" />
