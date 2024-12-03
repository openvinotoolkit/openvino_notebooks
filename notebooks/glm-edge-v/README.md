## Visual-language assistant with GLM-Edge-V and OpenVINO

The [GLM-Edge](https://huggingface.co/collections/THUDM/glm-edge-6743283c5809de4a7b9e0b8b) series is [Zhipu](https://huggingface.co/THUDM)'s attempt to meet real-world deployment scenarios for edge devices. It consists of two sizes of large language dialogue models and multimodal understanding models (GLM-Edge-1.5B-Chat, GLM-Edge-4B-Chat, GLM-Edge-V-2B, GLM-Edge-V-5B). Among them, the 1.5B / 2B models are mainly targeted at platforms like mobile phones and car machines, while the 4B / 5B models are aimed at platforms like PCs. Based on the technological advancements of the GLM-4 series, some targeted adjustments have been made to the model structure and size, balancing model performance, real-world inference efficiency, and deployment convenience. Through deep collaboration with partner enterprises and relentless efforts in inference optimization, the GLM-Edge series models can run at extremely high speeds on some edge platforms.

In this tutorial we consider how to launch multimodal model GLM-Edge-V using OpenVINO for creation multimodal chatbot. Additionally, we optimize model to low precision using [NNCF](https://github.com/openvinotoolkit/nncf)

![image](https://github.com/user-attachments/assets/06c51867-0580-4434-962e-31b6068c2001)

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

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/phi-3-vision/README.md" />

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/glm-edge-v/README.md" />
