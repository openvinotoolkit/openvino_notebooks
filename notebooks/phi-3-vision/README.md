## Visual-language assistant with Phi3-Vision and OpenVINO

The [Phi-3-Vision-128K-Instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision. The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures. More details about model can be found in [model blog post](https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/), [technical report](https://aka.ms/phi3-tech-report), [Phi-3-cookbook](https://github.com/microsoft/Phi-3CookBook)

In this tutorial we consider how to use Phi-3-Vision model to build multimodal chatbot using [Optimum Intel](https://github.com/huggingface/optimum-intel). Additionally, we optimize model to low precision using [NNCF](https://github.com/openvinotoolkit/nncf)

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
