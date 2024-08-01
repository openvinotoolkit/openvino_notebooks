# Create a RAG system using OpenVINO and LlamaIndex

**Retrieval-augmented generation (RAG)** is a technique for augmenting LLM knowledge with additional, often private or real-time, data. LLMs can reason about wide-ranging topics, but their knowledge is limited to the public data up to a specific point in time that they were trained on. If you want to build AI applications that can reason about private data or data introduced after a model’s cutoff date, you need to augment the knowledge of the model with the specific information it needs. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).

![UI](https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/5d145de6-69ee-4f7c-bfcb-6bdf6a5534a6)

[LlamaIndex](https://docs.llamaindex.ai/en/stable/) is a framework for building context-augmented generative AI applications with LLMs.LlamaIndex imposes no restriction on how you use LLMs. You can use LLMs as auto-complete, chatbots, semi-autonomous agents, and more. It just makes using them easier. 


## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert the model from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino).
- Compress model weights to 4-bit or 8-bit data types using [NNCF](https://github.com/openvinotoolkit/nncf)
- Create a RAG chain pipeline
- Run Q&A pipeline

## Installation Instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md)

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-rag-llamaindex/README.md" />
