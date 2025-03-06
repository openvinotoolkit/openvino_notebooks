## Visual-language assistant with GLM-Edge-V and OpenVINO

[GLM-4V-9B](https://huggingface.co/THUDM/glm-4v-9b) is an open source multimodal version of the GLM-4 series launched by Zhipu AI. GLM-4V-9B has the ability to conduct multi-round conversations in Chinese and English at a high resolution of 1120 * 1120. In multimodal evaluations of comprehensive Chinese and English abilities, perceptual reasoning, text recognition, and chart understanding, GLM-4V-9B has shown superior performance many popular models.

You can find more information in [model card](https://huggingface.co/THUDM/glm-4v-9b), [technical report](https://arxiv.org/pdf/2406.12793) and original [repository](https://github.com/THUDM/GLM-4).

In this tutorial we consider how to launch multimodal model GLM-Edge-V using OpenVINO for creation multimodal chatbot. Additionally, we optimize model to low precision using [NNCF](https://github.com/openvinotoolkit/nncf).

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
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/glm4-v/README.md" />
