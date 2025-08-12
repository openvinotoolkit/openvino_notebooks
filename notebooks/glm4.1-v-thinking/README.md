# Visual-language assistant with GLM-4.1V-9B-Thinking and OpenVINO

[GLM-4.1V-9B-Thinking](https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking) is based on the GLM-4-9B-0414 foundation model, designed to explore the upper limits of reasoning in vision-language models. By introducing a "thinking paradigm" and leveraging reinforcement learning, the model significantly enhances its capabilities. It achieves state-of-the-art performance among 10B-parameter VLMs, matching or even surpassing the 72B-parameter Qwen-2.5-VL-72B on 18 benchmark tasks. We are also open-sourcing the base model GLM-4.1V-9B-Base to support further research into the boundaries of VLM capabilities.

You can find more information in [model card](https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking) and [technical report](https://arxiv.org/abs/2507.01006)。

In this tutorial we consider how to launch multimodal model GLM-4.1V-9B-Thinking using OpenVINO for creation multimodal chatbot. Additionally, we optimize model to low precision using [NNCF](https://github.com/openvinotoolkit/nncf).

<img width="1938" height="924" alt="Image" src="https://github.com/user-attachments/assets/fd7287f2-d648-495c-84a3-f4cb8c4057bb" />

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/glm4.1-v-thinking/README.md" />
