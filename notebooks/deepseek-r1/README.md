# LLM reasoning with DeepSeek-R1 distilled models

[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) is an open-source reasoning model developed by DeepSeek to address tasks requiring logical inference, mathematical problem-solving, and real-time decision-making. With DeepSeek-R1, you can follow its logic, making it easier to understand and, if necessary, challenge its output. This capability gives reasoning models an edge in fields where outcomes need to be explainable, like research or complex decision-making.

Distillation in AI creates smaller, more efficient models from larger ones, preserving much of their reasoning power while reducing computational demands. DeepSeek applied this technique to create a suite of distilled models from R1, using Qwen and Llama architectures. That allows us to try DeepSeek-R1 capability locally on usual laptops.

In this tutorial, we consider how to run DeepSeek-R1 distilled models using OpenVINO.

The tutorial supports different models, you can select one from the provided options to compare the quality of LLM solutions:

* **DeepSeek-R1-Distill-Llama-8B** is a distilled model based on [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), that prioritizes high performance and advanced reasoning capabilities, particularly excelling in tasks requiring mathematical and factual precision. Check [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) for more info.
* **DeepSeek-R1-Distill-Qwen-1.5B** is the smallest DeepSeek-R1 distilled model based on [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B). Despite its compact size, the model demonstrates strong capabilities in solving basic mathematical tasks, at the same time its programming capabilities are limited. Check [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) for more info.
* **DeepSeek-R1-Distill-Qwen-7B** is a distilled model based on [Qwen-2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B). The model demonstrates a good balance between mathematical and factual reasoning and can be less suited for complex coding tasks. Check [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) for more info.
* **DeepSeek-R1-Distil-Qwen-14B** is a distilled model based on [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B) that has great competence in factual reasoning and solving complex mathematical tasks.  Check [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) for more info.

Learn how to accelerate **DeepSeek-R1-Distill-Llama-8B** with **FastDraft** and OpenVINO GenAI speculative decoding pipeline in this [notebook](../../supplementary_materials/notebooks/fastdraft-deepseek/fastdraft_deepseek.ipynb)
## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert the model from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino).
- Compress model weights to INT4 or INT8 precision using [NNCF](https://github.com/openvinotoolkit/nncf)
- Create an inference pipeline
- Run interactive demo
  
![](https://github.com/user-attachments/assets/9062bdc4-0338-4555-a863-87b5a71236e9)

## Installation Instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/deepseek-r1/README.md" />
