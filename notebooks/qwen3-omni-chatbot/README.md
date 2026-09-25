# Omnimodal assistant with Qwen3-Omni and OpenVINO

Qwen3-Omni is the natively end-to-end multilingual omni-modal foundation models. It processes text, images, audio, and video, and delivers real-time streaming responses in both text and natural speech. We introduce several architectural upgrades to improve performance and efficiency. Key features:

* **State-of-the-art across modalities**: Early text-first pretraining and mixed multimodal training provide native multimodal support. While achieving strong audio and audio-video results, unimodal text and image performance does not regress. Reaches SOTA on 22 of 36 audio/video benchmarks and open-source SOTA on 32 of 36; ASR, audio understanding, and voice conversation performance is comparable to Gemini 2.5 Pro.

* **Multilingual**: Supports 119 text languages, 19 speech input languages, and 10 speech output languages.
  - **Speech Input**: English, Chinese, Korean, Japanese, German, Russian, Italian, French, Spanish, Portuguese, Malay, Dutch, Indonesian, Turkish, Vietnamese, Cantonese, Arabic, Urdu.
  - **Speech Output**: English, Chinese, French, German, Russian, Italian, Spanish, Portuguese, Japanese, Korean.

* **Novel Architecture**: MoE-based Thinker–Talker design with AuT pretraining for strong general representations, plus a multi-codebook design that drives latency to a minimum.

* **Real-time Audio/Video Interaction**: Low-latency streaming with natural turn-taking and immediate text or speech responses.

* **Flexible Control**: Customize behavior via system prompts for fine-grained control and easy adaptation.

* **Detailed Audio Captioner**: Qwen3-Omni-30B-A3B-Captioner is now open source: a general-purpose, highly detailed, low-hallucination audio captioning model that fills a critical gap in the open-source community.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/q3o_introduction.png" width="90%"/>
<p>

More details about model can be found in [model card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) and original [repo](https://github.com/QwenLM/Qwen3-Omni/tree/main).

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Download PyTorch model
- Convert model to OpenVINO Intermediate Representation (IR)
- Compress Language Model weights
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content. Image bellow shows a result of model work.
![Image](https://github.com/user-attachments/assets/83e1e0f7-1a12-426b-b3f8-794662812cd4)


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
