# Text-to-Speech synthesis using Kokoro and OpenVINO GenAI

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers high-quality speech while remaining fast and resource-efficient.

This tutorial demonstrates end-to-end Kokoro inference with [`openvino_genai.Text2SpeechPipeline`](https://github.com/openvinotoolkit/openvino.genai). It supports FP16, INT8, and INT4 weights. The ready-to-run [OpenVINO/Kokoro-82M-int8-ov](https://huggingface.co/OpenVINO/Kokoro-82M-int8-ov) model is downloaded by default; formats unavailable on Hugging Face Hub are exported locally with Optimum Intel.

## Notebook Contents

The tutorial consists of the following steps:

* Select FP16, INT8, or INT4 weights
* Download a preconverted model when available or export Kokoro locally
* Select an OpenVINO inference device
* Load a Kokoro voice embedding
* Run text-to-speech synthesis with OpenVINO GenAI
* Interactive demo

Inference with the preconverted model supports Python 3.10 and newer, including Python 3.13. Local export requires Python 3.10–3.12 because the `kokoro` and `misaki` packages currently require Python `<3.13`.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

The notebook installs `espeakng-loader` and configures its bundled `espeak-ng` library. OpenVINO GenAI uses it as a fallback for unknown English words and as the primary text-to-phoneme engine for supported non-English languages.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/kokoro/README.md" />
