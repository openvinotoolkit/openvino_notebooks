# Visual-language assistant with InternVL2 and OpenVINO

InternVL 2.0 is the latest addition to the InternVL series of multimodal large language models. InternVL 2.0 features a variety of instruction-tuned models, ranging from 1 billion to 108 billion parameters. Compared to the state-of-the-art open-source multimodal large language models, InternVL 2.0 surpasses most open-source models. It demonstrates competitive performance on par with proprietary commercial models across various capabilities, including document and chart comprehension, infographics QA, scene text understanding and OCR tasks, scientific and mathematical problem solving, as well as cultural understanding and integrated multimodal capabilities.

More details about model can be found in [model card](https://huggingface.co/OpenGVLab/InternVL2-4B), [blog](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) and original [repo](https://github.com/OpenGVLab/InternVL).

In this tutorial we consider how to convert and optimize InternVL2 model for creating multimodal chatbot. Additionally, we demonstrate how to apply stateful transformation on LLM part and model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf)

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.

The image bellow illustrates example of input prompt and model answer.
![example.png](https://github.com/user-attachments/assets/1c3cf42a-db40-4fa2-81e0-ead9bde7ace6)

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/internvl2/README.md" />
