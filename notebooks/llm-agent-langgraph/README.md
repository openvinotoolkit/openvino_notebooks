# Self-Correcting RAG Agent with LangGraph and OpenVINO

Standard Retrieval-Augmented Generation (RAG) pipelines follow a linear path: retrieve documents, then generate an answer. If retrieval returns irrelevant documents, the model often hallucinates an answer instead of acknowledging uncertainty.

This notebook demonstrates a **self-correcting RAG agent** powered by [LangGraph](https://langchain-ai.github.io/langgraph/) and [OpenVINO](https://docs.openvino.ai/). The agent implements a stateful workflow with conditional branching:

1. **Retrieve** documents from a ChromaDB vector store
2. **Grade** each document for relevance using an OpenVINO-optimized LLM
3. **Generate** an answer if relevant documents are found
4. **Rewrite** the query and re-retrieve if documents are irrelevant

All LLM inference is accelerated with OpenVINO on Intel CPUs, GPUs, and NPUs.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Select and convert a model (Phi-3-mini, Llama-3.2-1B, or Qwen2.5-1.5B) to OpenVINO IR with INT4 compression
- Build a knowledge base with document chunking and ChromaDB
- Define the LangGraph agent with four nodes: retrieve, grade, generate, rewrite
- Run the self-correcting agent on sample queries
- Interactive demo with Gradio UI showing agent reasoning

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-agent-langgraph/README.md" />
