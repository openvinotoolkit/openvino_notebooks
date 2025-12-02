# Multi-speaker dialogue generation with FireRedTTS‑2 and OpenVINO

FireRedTTS‑2 is a long-form streaming TTS system for multi-speaker dialogue generation, delivering stable, natural speech with reliable speaker switching and context-aware prosody. It is highlighted by following features:
- **Long Conversational Speech Generation**: It currently supports 3 minutes dialogues with 4 speakers and can be easily scaled to longer conversations
with more speakers by extending training corpus.
- **Multilingual Support**: It supports multiple languages including English, Chinese, Japanese, Korean, French, German, and Russian. Support zero-shot voice cloning for cross-lingual and code-switching scenarios.
- **Ultra-Low Latency**: Building on the new **12.5Hz streaming** speech tokenizer, we employ a dual-transformer architecture that operates on a text–speech interleaved sequence, enabling flexible sentence-by-sentence generation and reducing first-packet latency，Specifically, on an L20 GPU, our first-packet latency as low as 140ms while maintaining high-quality audio output.
- **Strong Stability**：Our model achieves high similarity and low WER/CER in both monologue and dialogue tests.
- **Random Timbre Generation**:Useful for creating ASR/speech interaction data.

More details can be found in the [paper](https://arxiv.org/abs/2509.02020), original [repository](https://github.com/FireRedTeam/FireRedTTS2) and [model card](https://huggingface.co/FireRedTeam/FireRedTTS2)

In this tutorial we consider how to run and optimize FireRedTTS‑2 using OpenVINO.

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create an interactive assistant that can generate multi-speaker dialogues, perform voice cloning, and synthesize natural speech using FireRedTTS-2 and OpenVINO.

The images bellow illustrates example of voice cloning and dialogue generation.

<img width="1862" height="1125" alt="image" src="https://github.com/user-attachments/assets/a7512db5-78cd-4379-956b-893c13534862" />

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/fireredtts2/README.md" />
