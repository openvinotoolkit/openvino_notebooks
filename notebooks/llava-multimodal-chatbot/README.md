# Visual-language assistant with LLaVA and OpenVINO

![llava_logo.png](https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/llava_logo.png)

*image source: [LLaVA repository](https://github.com/haotian-liu/LLaVA/blob/main/images/llava_logo.png)*

[LLaVA](https://llava-vl.github.io) (Large Language and Vision Assistant) is large multimodal model that aims to develop a general-purpose visual assistant that can follow both language and image instructions to complete various real-world tasks. The idea is to combine the power of large language models (LLMs) with vision encoders like CLIP to create an end-to-end trained neural assistant that understands and acts upon multimodal instructions.

While LLaVA excels at image-based tasks, Video-LLaVA expands this fluency to the dynamic world of videos, enabling seamless comprehension and reasoning across both visual domains. This means it can answer questions, generate text, and perform other tasks with equal ease, regardless of whether it's presented with a still image or a moving scene.

In the field of artificial intelligence, the goal is to create a versatile assistant capable of understanding and executing tasks based on both visual and language inputs. Current approaches often rely on large vision models that solve tasks independently, with language only used to describe image content. While effective, these models have fixed interfaces with limited interactivity and adaptability to user instructions. On the other hand, large language models (LLMs) have shown promise as a universal interface for general-purpose assistants. By explicitly representing various task instructions in language, these models can be guided to switch and solve different tasks. To extend this capability to the multimodal domain, the [LLaVA paper](https://arxiv.org/abs/2304.08485) introduces  `visual instruction-tuning`, a novel approach to building a general-purpose visual assistant. 

In this tutorial series we consider how to use LLaVA and Video-LLaVA model to build multimodal chatbot with OpenVINO help.

## LLaVA
### Notebook contents
The tutorial consists from following steps:

- Install prerequisites
- Prepare input processor and tokenizer
- Download original model
- Compress model weights to 4 and 8 bits using NNCF
- Convert model to OpenVINO Intermediate Representation (IR) format
- Prepare OpenVINO-based inference pipeline
- Run OpenVINO model

### Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

## Video-LLaVA
### Notebook contents
The tutorial consists from following steps:

- Install prerequisites
- Download original model
- Compress model weights to 4 and 8 bits using NNCF
- Convert model to OpenVINO Intermediate Representation (IR) format
- Prepare OpenVINO-based inference pipeline
- Run OpenVINO model

### Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).