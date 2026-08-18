# Muse Glimmer-30B multimodal reasoning with OpenVINO™

Muse Glimmer-30B is a dense multimodal model from Meta Superintelligence Lab designed for local agentic workloads. It combines a causal language model with a dedicated perception encoder and supports image understanding, video understanding, code generation, tool use, and controllable multi-step reasoning.

This tutorial demonstrates how to:

- convert and compress [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) with Optimum Intel and NNCF;
- run image and video understanding with the OpenVINO GenAI `VLMPipeline`;
- separate the model's ATEM reasoning channel from its final response;
- use visual coding to recreate a playable browser game from a reference image;
- generate a playable browser game from a text-only prompt;
- launch a multimodal Gradio chat that presents reasoning and answers separately.

Muse Glimmer requires `transformers==5.15`, a recent Optimum Intel build, and recent OpenVINO, OpenVINO Tokenizers, and OpenVINO GenAI packages. The notebook uses INT4 asymmetric compression by default because the original model has approximately 29.6 billion parameters.

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to the [Installation Guide](../../README.md#-installation-guide).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/muse-glimmer/README.md" />
