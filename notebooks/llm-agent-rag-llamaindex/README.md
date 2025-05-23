# Create an Agentic RAG using OpenVINO and LlamaIndex


An **agent** is an automated reasoning and decision engine. It takes in a user input/query and can make internal decisions for executing that query in order to return the correct result. The key agent components can include, but are not limited to:

- Breaking down a complex question into smaller ones
- Choosing an external Tool to use + coming up with parameters for calling the Tool
- Planning out a set of tasks
- Storing previously completed tasks in a memory module

[LlamaIndex](https://docs.llamaindex.ai/en/stable/) is a framework for building context-augmented generative AI applications with LLMs.LlamaIndex imposes no restriction on how you use LLMs. You can use LLMs as auto-complete, chatbots, semi-autonomous agents, and more. It just makes using them easier. You can build agents on top of your existing LlamaIndex RAG pipeline to empower it with automated decision capabilities. A lot of modules (routing, query transformations, and more) are already agentic in nature in that they use LLMs for decision making.

![agentic-rag](https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/871cb90d-27fd-4a87-aa3c-f4cdb199a148)


This example will demonstrate using RAG engines as a tool in an agent with OpenVINO and LlamaIndex.


### Notebook Contents

The tutorial consists of the following steps:

- Prerequisites
- Create tools
- Create prompt template
- Create LLM
  - Download model
  - Select inference device for LLM
- Create agent
- Run the agent
- Interactive Demo
  - Use built-in tool
  - Create customized tools
  - Create AI agent demo with Gradio UI

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-agent-rag-llamaindex/README.md" />
