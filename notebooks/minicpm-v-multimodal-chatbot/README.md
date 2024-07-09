# Visual-language assistant with MiniCPM-V2 and OpenVINO

MiniCPM-V 2 is a strong multimodal large language model for efficient end-side deployment. The model is built based on SigLip-400M and MiniCPM-2.4B, connected by a perceiver resampler. MiniCPM-V 2.0 has several notable features:
* **Outperforming many popular models on many benchmarks** (including OCRBench, TextVQA, MME, MMB, MathVista, etc). Strong OCR capability, achieving comparable performance to Gemini Pro in scene-text understanding.
* **Trustworthy Behavior**. LMMs are known for suffering from hallucination, often generating text not factually grounded in images. MiniCPM-V 2.0 is the first end-side LMM aligned via multimodal RLHF for trustworthy behavior (using the recent [RLHF-V](https://rlhf-v.github.io/) [CVPR'24] series technique). This allows the model to match GPT-4V in preventing hallucinations on Object HalBench.
* **High-Resolution Images at Any Aspect Raito.** MiniCPM-V 2.0 can accept 1.8 million pixels (e.g., 1344x1344) images at any aspect ratio. This enables better perception of fine-grained visual information such as small objects and optical characters, which is achieved via a recent technique from [LLaVA-UHD](https://arxiv.org/pdf/2403.11703).
* **High Efficiency.** For visual encoding, model compresses the image representations into much fewer tokens via a perceiver resampler. This allows MiniCPM-V 2.0 to operate with favorable memory cost and speed during inference even when dealing with high-resolution images.
* **Bilingual Support.** MiniCPM-V 2.0 supports strong bilingual multimodal capabilities in both English and Chinese. This is enabled by generalizing multimodal capabilities across languages, a technique from [VisCPM](https://arxiv.org/abs/2308.12038)[ICLR'24].

In this tutorial we consider how to convert and optimize MiniCPM-V2 model for creating multimodal chatbot. Additionally, we demonstrate how to apply stateful transformation on LLM part and model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf) 

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Download PyTorch model
- Convert model to OpenVINO Intermediate Representation (IR)
- Compress Language Model weights
- Prepare Inference Pipeline
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content. Image bellow shows a result of model work.
![](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/2727402e-3697-442e-beca-26b149967c84)


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).