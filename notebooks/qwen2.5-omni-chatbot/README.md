# Omnimodal assistant with Qwen2.5-Omni and OpenVINO

Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner.

![Image](https://github.com/user-attachments/assets/600798db-c80e-4d1d-ab00-fa945cdcd583)

More details about model can be found in [model card](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) and original [repo](https://github.com/QwenLM/Qwen2.5-Omni).

In this tutorial we consider how to convert and optimize Qwen2.5-Omni model for creating omnimodal chatbot. Additionally, we demonstrate how to apply stateful transformation on LLM part and model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf)

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Download PyTorch model
- Convert model to OpenVINO Intermediate Representation (IR)
- Compress Language Model weights
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content. Image bellow shows a result of model work.
![Image](https://github.com/user-attachments/assets/83e1e0f7-1a12-426b-b3f8-794662812cd4)


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen2.5-omni-chatbot/README.md" />
