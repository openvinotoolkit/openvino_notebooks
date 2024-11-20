# Create a ReAct Agent using OpenVINO

LLM are limited to the knowledge on which they have been trained and the additional knowledge provided as context, as a result, if a useful piece of information is missing the provided knowledge, the model cannot “go around” and try to find it in other sources. This is the reason why we need to introduce the concept of Agents.

The core idea of agents is to use a language model to choose a sequence of actions to take. In agents, a language model is used as a reasoning engine to determine which actions to take and in which order. Agents can be seen as applications powered by LLMs and integrated with a set of tools like search engines, databases, websites, and so on. Within an agent, the LLM is the reasoning engine that, based on the user input, is able to plan and execute a set of actions that are needed to fulfill the request.

![image](https://github.com/user-attachments/assets/b656adab-a448-4784-a6df-a068e0cb45bb)

This notebook explores how to create an ReAct Agent step by step using OpenVINO. [ReAct](https://arxiv.org/abs/2210.03629) is an approach to combine reasoning (e.g. chain-of-thought prompting) and acting. ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. 


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

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-agent-react/README.md" />
