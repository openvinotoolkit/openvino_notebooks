# End-to-End Speech Recognition with Fun-ASR-Nano and OpenVINO

Fun-ASR is an end-to-end speech recognition large model launched by Tongyi Lab. It is trained on tens of millions of hours of real speech data, possessing powerful contextual understanding capabilities and industry adaptability. It supports low-latency real-time transcription and covers 31 languages. It excels in vertical domains such as education and finance, accurately recognizing professional terminology and industry expressions, effectively addressing challenges like "hallucination" generation and language confusion, achieving "clear hearing, understanding meaning, and accurate writing."

<img width="792" height="479" alt="image" src="https://github.com/user-attachments/assets/d55ea91b-0dd2-4a92-b6a1-3460edb41b6f" />

More details can be found in the original [repository](https://github.com/FunAudioLLM/Fun-ASR) and [model card](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)


### Notebook Contents

In this tutorial we consider how to run and optimize Fun-ASR-Nano using OpenVINO.

The tutorial consists of the following steps:

- Install prerequisites
- Convert model to OpenVINO intermediate representation (IR) format 
- Prepare OpenVINO Inference pipeline
- Run Speech Recognition
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/funasr-nano/README.md" />
