# Video Classification with InternVideo2 and OpenVINO

InternVideo2 is family of video foundation models (ViFM) that achieve the state-of-the-art results in video recognition, video-text tasks, and video-centric dialogue.
You can find more information about model in [model card](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_6B), [paper](https://arxiv.org/pdf/2403.15377) and original [repository](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2/multi_modality).

In this tutorial we consider how to convert, optimize and run InternVideo2 Stage2 model for video classification using OpenVINO.

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create text-to-video retrieval pipeline which is responsible to find the most suitable text caption for video content.

The image bellow illustrates example of model inference result.
![example.png](https://github.com/user-attachments/assets/6720efe0-ab24-4d73-a22f-a8a0499558d8)

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/intern-video2-classiciation/README.md" />
