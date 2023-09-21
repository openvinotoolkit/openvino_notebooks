# Text-to-Speech generation with BARK and OpenVINO

![bark_generated.png](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/9a770279-0045-480e-95f2-1a2f2d0a5115)

*image generated using [Stable Diffusion XL notebook](../248-stable-diffusion-xl/)*

🐶 Bark is a transformer-based text-to-audio model created by [Suno](https://suno.ai). Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying. 

With Bark, users can also produce nonverbal communications like laughing, sighing, and crying, making it a versatile tool for a variety of applications.

![image.png](https://user-images.githubusercontent.com/5068315/235310676-a4b3b511-90ec-4edf-8153-7ccf14905d73.png)

Bark is a cutting-edge text-to-speech (TTS) technology that has taken the AI world by storm. Unlike the typical TTS engines that sound robotic and mechanic, Bark offers human-like voices that are highly realistic and natural sounding. Bark uses GPT-style models to generate speech with minimal tweaking, producing highly expressive and emotive voices that can capture nuances such as tone, pitch, and rhythm. It offers a fantastic experience that can leave you wondering if you’re listening to human beings.

Notably, Bark supports multiple languages and can generate speech in Mandarin, French, Italian, Spanish, and other languages with impressive clarity and accuracy. With Bark, you can easily switch between languages and still enjoy high-quality sound effects.

Bark is not only intelligent but also intuitive, making it an ideal tool for individuals and businesses looking to create high-quality voice content for their platforms. Whether you’re looking to create podcasts, audiobooks, video game sounds, or any other form of voice content, Bark has you covered.

So, if you’re looking for a revolutionary text-to-speech technology that can elevate your voice content, Bark is the way to go!
In this tutorial we consider how to convert and run bark with OpenVINO.

## Notebook contents
The tutorial consists of the following steps:

- Install and import required packages
- Load the PyTorch model
- Convert the PyTorch model
- Build a pipeline
- Inference with OpenVINO
- Run Interactive demo

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).