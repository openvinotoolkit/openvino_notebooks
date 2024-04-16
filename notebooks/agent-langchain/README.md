# Create an AI Agent using OpenVINO

LLM are limited to the knowledge on which they have been trained and the additional knowledge provided as context, as a result, if a useful piece of information is missing the provided knowledge, the model cannot “go around” and try to find it in other sources. This is the reason why we need to introduce the concept of Agents.

The core idea of agents is to use a language model to choose a sequence of actions to take. In agents, a language model is used as a reasoning engine to determine which actions to take and in which order. Agents can be seen as applications powered by LLMs and integrated with a set of tools like search engines, databases, websites, and so on. Within an agent, the LLM is the reasoning engine that, based on the user input, is able to plan and execute a set of actions that are needed to fulfill the request.

![agent ui](https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/2abb2389-e612-4599-82c6-64cdac259120)

This notebook explores how to create an AI Agent step by step using OpenVINO and LangChain. [LangChain](https://python.langchain.com/docs/get_started/introduction) is a framework for developing applications powered by language models. It comes with a number of built-in agents that are optimized for different use cases.

LLM models can be run locally through the `HuggingFacePipeline` class in LangChain. To deploy a model with OpenVINO, you can specify the `backend="openvino"` parameter to trigger OpenVINO as backend inference framework. For [more information](https://python.langchain.com/docs/integrations/llms/openvino/).


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

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
