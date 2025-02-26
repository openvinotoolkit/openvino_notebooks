# Multimodal assistant with Phi-4-mini-multimodal and OpenVINO

Phi-4-multimodal-instruct is a lightweight open multimodal foundation model. The model processes text, image, and audio inputs, generating text outputs. Phi-4-multimodal-instruct has 5.6B parameters and is a multimodal transformer model. The model has the pretrained Phi-4-mini as the  backbone language model, and the advanced encoders and adapters of vision and speech.
In this tutorial we will explore how to run Phi-4-mini-multimodal model using [OpenVINO](https://github.com/openvinotoolkit/openvino) and optimize it using [NNCF](https://github.com/openvinotoolkit/nncf).

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's and audio's content.
![phi4](https://github.com/user-attachments/assets/8c0b8e50-417e-4579-b799-e9b9c15e8a87)

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/phi-4-multimodal/README.md" />
