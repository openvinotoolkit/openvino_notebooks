# Visual-language assistant using DeepSeek-VL2 and OpenVINO

DeepSeek-VL2 is an advanced series of large Mixture-of-Experts (MoE) Vision-Language models. . DeepSeek-VL2 demonstrates superior capabilities across various tasks, including but not limited to visual question answering, optical character recognition, document/table/chart understanding, and visual grounding.

More details can be found in the [paper](https://arxiv.org/abs/2412.10302) and original [repository](https://github.com/deepseek-ai/DeepSeek-VL2).

In this tutorial we consider how to convert and run DeepSeek-VL2 models using [OpenVINO](https://github.com/openvinotoolkit/openvino) and optimize it using [NNCF](https://github.com/openvinotoolkit/nncf).

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.

![cat_deepseek](https://github.com/user-attachments/assets/343c2906-b4ec-4191-8ef4-375d04005cf0)


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/deepseek-vl2/README.md" />
