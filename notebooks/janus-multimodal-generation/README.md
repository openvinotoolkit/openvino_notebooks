# Multimodal understanding and generation with Janus and OpenVINO

Janus is a novel autoregressive framework that unifies multimodal understanding and generation. It addresses the limitations of previous approaches by decoupling visual encoding into separate pathways, while still utilizing a single, unified transformer architecture for processing. The decoupling not only alleviates the conflict between the visual encoder’s roles in understanding and generation, but also enhances the framework’s flexibility. Janus surpasses previous unified model and matches or exceeds the performance of task-specific models. The simplicity, high flexibility, and effectiveness of Janus make it a strong candidate for next-generation unified multimodal models.

More details can be found in the [paper](https://arxiv.org/abs/2410.13848), original [repository](https://github.com/deepseek-ai/Janus) and [model card](https://huggingface.co/deepseek-ai/Janus-1.3B)

In this tutorial we consider how to run and optimize Janus using OpenVINO. Additionally, we demonstrate how to apply stateful transformation on LLM part and model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf)

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive assistant that can answer questions about provided image's content or generate images based on text instructions.

The images bellow illustrates example of input prompt and model answer for image understanding and generation
![example.png](https://github.com/user-attachments/assets/89a71be8-b472-4acd-a2e0-dbc97645fc1c)
![example2.png](https://github.com/user-attachments/assets/5aca2b37-52d9-403d-a773-311ccf82b375)

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/janus-multimodal-generation/README.md" />
