# LLM reasoning with Deepseek-R1 distilled models

[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) is an open-source reasoning model developed by DeepSeek to address tasks requiring logical inference, mathematical problem-solving, and real-time decision-making. With DeepSeek-R1, you can follow its logic, making it easier to understand and, if necessary, challenge its output. This capability gives reasoning models an edge in fields where outcomes need to be explainable, like research or complex decision-making.

Distillation in AI creates smaller, more efficient models from larger ones, preserving much of their reasoning power while reducing computational demands. DeepSeek applied this technique to create a suite of distilled models from R1, using Qwen and Llama architectures. That allows us to try Deepseek-R1 capability locally on usual laptops.

In this tutorial, we consider how to run Deepseek-R1 distilled models using OpenVINO.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert the model from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino).
- Compress model weights to INT4 or INT8 precision using [NNCF](https://github.com/openvinotoolkit/nncf)
- Create an inference pipeline
- Run interactive demo

## Installation Instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/deepseek-r1/README.md" />
