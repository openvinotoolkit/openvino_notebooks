# Text-to-speech (TTS) with Parler-TTS and OpenVINO™

Parler-TTS is a lightweight text-to-speech (TTS) model that can generate high-quality, natural sounding speech in the style of a given speaker (gender, pitch, speaking style, etc). It is a reproduction of work from the paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://www.text-description-to-speech.com/) by Dan Lyth and Simon King, from Stability AI and Edinburgh University respectively.

![](https://images.squarespace-cdn.com/content/v1/657816dfbefe0533e8a69d9a/30c96e25-acc5-4019-acdd-648da6142c4c/architecture_v3.png?format=2500w)

Text-to-speech models trained on large-scale datasets have demonstrated impressive in-context learning capabilities and naturalness. However, control of speaker identity and style in these models typically requires conditioning on reference speech recordings, limiting creative applications. Alternatively, natural language prompting of speaker identity and style has demonstrated promising results and provides an intuitive method of control. However, reliance on human-labeled descriptions prevents scaling to large datasets.

This work bridges the gap between these two approaches. The authors propose a scalable method for labeling various aspects of speaker identity, style, and recording conditions. This method then is applied to a 45k hour dataset, which is used to train a speech language model. Furthermore, the authors propose simple methods for increasing audio fidelity, significantly outperforming recent work despite relying entirely on found data.


[GitHub repository](https://github.com/huggingface/parler-tts)

[HuggingFace page](https://huggingface.co/parler-tts)


## Notebook Contents

This notebook demonstrates how to convert and run the Parler TTS model using OpenVINO.

Notebook contains the following steps:
1. Load the original model and inference.
2. Convert the model to OpenVINO IR.
3. Compiling models and inference.
4. Interactive inference.

## Installation instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/parler-tts-text-to-speech/README.md" />
