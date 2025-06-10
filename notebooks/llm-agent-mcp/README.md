# Create MCP Agent using OpenVINO and Qwen-Agent

MCP is an open protocol that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect your devices to various peripherals and accessories, MCP provides a standardized way to connect AI models to different data sources and tools.

MCP helps you build agents and complex workflows on top of LLMs. LLMs frequently need to integrate with data and tools, and MCP provides:

- A growing list of pre-built integrations that your LLM can directly plug into
- The flexibility to switch between LLM providers and vendors
- Best practices for securing your data within your infrastructure

![Image](https://github.com/user-attachments/assets/bdb1790a-b464-457b-ae57-a88b9f9bae43)

[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) is a framework for developing LLM applications based on the instruction following, tool usage, planning, and memory capabilities of Qwen. It also comes with example applications such as Browser Assistant, Code Interpreter, and Custom Assistant.

This notebook explores how to create a MCP Agent step by step using OpenVINO and Qwen-Agent.

### Notebook Contents

The tutorial consists of the following steps:

- Prerequisites
- Convert model using Optimum-CLI tool
- Create an Agent
- Interactive Demo


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-agent-mcp/README.md" />
