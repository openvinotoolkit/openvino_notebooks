# Text-to-video generation with ZeroScope and OpenVINO

|Darth Vader is surfing on waves|
| :---: |
|![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/darthvader_cerpense.gif)|

The ZeroScope model is a free and open-source text-to-video model that can generate realistic and engaging videos from text descriptions. It is based on the [Modelscope](https://modelscope.cn/models/damo/text-to-video-synthesis/summary) model, but it has been improved to produce higher-quality videos with a 16:9 aspect ratio and no Shutterstock watermark. The ZeroScope model is available in two versions: ZeroScope_v2 576w, which is optimized for rapid content creation at a resolution of 576x320 pixels, and ZeroScope_v2 XL, which upscales videos to a high-definition resolution of 1024x576.

The ZeroScope model is trained on a dataset of over 9,000 videos and 29,000 tagged frames. It uses a diffusion model to generate videos, which means that it starts with a random noise image and gradually adds detail to it until it matches the text description. The ZeroScope model is still under development, but it has already been used to create some impressive videos. For example, it has been used to create videos of people dancing, playing sports, and even driving cars.

The ZeroScope model is a powerful tool that can be used to create various videos, from simple animations to complex scenes. It is still under development, but it has the potential to revolutionize the way we create and consume video content.

Both versions of the ZeroScope model are available on Hugging Face:
- [ZeroScope_v2 576w](https://huggingface.co/cerspense/zeroscope_v2_576w)
- [ZeroScope_v2 XL](https://huggingface.co/cerspense/zeroscope_v2_XL)

We will use the first one.
## Notebook contents
The tutorial consists of the following steps:

- Install and import required packages
- Load the model
- Convert the model
- Build a pipeline
- Inference with OpenVINO

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).