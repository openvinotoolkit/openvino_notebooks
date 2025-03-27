# Text Generation via Prompt Lookup Decoding using OpenVINO™

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/prompt-lookup-decoding/prompt-lookup-decoding.ipynb)


As model sizes grow, Generative AI implementations require significant inference resources. This not only increases the cost per generation from a prompt, but also increases the power consumption used to serve such requests.

Inference optimizations for text generation are essential for reducing costs and power consumption. When optimizing the inference process, the amount of time and energy required to generate text can be significantly reduced. This can lead to cost savings in terms of hardware and software, as well as reduced power consumption. Additionally, inference optimizations can help improve the accuracy of text generation as well as the speed at which it can be generated. This can lead to an improved user experience and increased efficiency in text-generation tasks. In summary, inference optimizations for text generation are essential to reduce costs and power consumption, while also improving the accuracy and speed of text generation.

[Prompt Lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) is [assisted-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency) technique, that allows to speed up token generation, where the draft model is replaced with simple string matching the prompt to generate candidate token sequences. 

Prompt Lookup decoding works the following way. Input defines as all the tokens till the current generation step (input_ids). It then tries to match last few tokens to somewhere earlier in the prompt. If found, it returns the next-k token continuation as `candidate input ids` or `candidate sequence`.

![](https://blog.vllm.ai/assets/figures/spec-decode/figure3.png)

This method highly effective for input grounded generation (summarization, document QA, multi-turn chat, code editing), where there is high n-gram overlap between LLM input (prompt) and LLM output. This could be entity names, phrases, or code chunks that the LLM directly copies from the input while generating the output. Prompt lookup exploits this pattern to speed up autoregressive decoding in LLMs. This results in significant speedups with no effect on output quality.

In this tutorial we consider how to apply [Prompt Lookup decoding with OpenVINO GenAI](https://medium.com/openvino-toolkit/enhancing-llm-inference-with-prompt-lookup-decoding-and-openvino-genai-e15b69aeaeab).

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download models
- Run prompt lookup decoding example and compare speed-up with respect to autoregressive sampling.

## Installation instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/prompt-lookup-decoding/README.md" />