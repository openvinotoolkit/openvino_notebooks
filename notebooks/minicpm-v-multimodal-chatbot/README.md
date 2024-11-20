# Visual-language assistant with MiniCPM-V2 and OpenVINO

MiniCPM-V 2 is a strong multimodal large language model for efficient end-side deployment. MiniCPM-V 2.6 is the latest and most capable model in the MiniCPM-V series. The model is built on SigLip-400M and Qwen2-7B with a total of 8B parameters. It exhibits a significant performance improvement over previous versions, and introduces new features for multi-image and video understanding.

More details about model can be found in [model card](https://huggingface.co/openbmb/MiniCPM-V-2_6) and original [repo](https://github.com/OpenBMB/MiniCPM-V).

In this tutorial we consider how to convert and optimize MiniCPM-V2.6 model for creating multimodal chatbot. Additionally, we demonstrate how to apply stateful transformation on LLM part and model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf)


## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Download PyTorch model
- Convert model to OpenVINO Intermediate Representation (IR)
- Compress Language Model weights
- Prepare Inference Pipeline using OpenVINO GenAI
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content. Image bellow shows a result of model work.
![](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/7b0919ea-6fe4-4c8f-8395-cb0ee6e87937)


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/minicpm-v-multimodal-chatbot/README.md" />
