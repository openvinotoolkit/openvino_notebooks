# Create a Function-calling Agent using OpenVINO

LLM are limited to the knowledge on which they have been trained and the additional knowledge provided as context, as a result, if a useful piece of information is missing the provided knowledge, the model cannot “go around” and try to find it in other sources. This is the reason why we need to introduce the concept of Agents.

The core idea of agents is to use a language model to choose a sequence of actions to take. In agents, a language model is used as a reasoning engine to determine which actions to take and in which order. Agents can be seen as applications powered by LLMs and integrated with a set of tools like search engines, databases, websites, and so on. Within an agent, the LLM is the reasoning engine that, based on the user input, is able to plan and execute a set of actions that are needed to fulfill the request.

![agent ui](https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/202cddac-dbbb-493b-ae79-0d45798f75c1)

This notebook explores how to create Function-calling Agent step by step using OpenVINO. Function calling allows a model to detect when one or more tools should be called and respond with the inputs that should be passed to those tools. In an API call, you can describe tools and have the model intelligently choose to output a structured object like JSON containing arguments to call these tools. The goal of tools APIs is to more reliably return valid and useful tool calls than what can be done using a generic text completion or chat API.


### Notebook Contents

The tutorial consists of the following steps:

- Prerequisites
- Create a Function calling agent
  - Create functions
  - Download model
  - Select inference device for LLM
  - Create LLM for Qwen-Agent
  - Create Function-calling pipeline
- Interactive Demo
  - Create tools
  - Create AI agent demo with Qwen-Agent and Gradio UI


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-agent-functioncall/README.md" />
