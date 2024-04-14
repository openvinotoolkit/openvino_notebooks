# Create a RAG system using OpenVINO and LangChain

**Retrieval-augmented generation (RAG)** is a technique for augmenting LLM knowledge with additional, often private or real-time, data. LLMs can reason about wide-ranging topics, but their knowledge is limited to the public data up to a specific point in time that they were trained on. If you want to build AI applications that can reason about private data or data introduced after a model’s cutoff date, you need to augment the knowledge of the model with the specific information it needs. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).

[LangChain](https://python.langchain.com/docs/get_started/introduction) is a framework for developing applications powered by language models. It has a number of components specifically designed to help build RAG applications. In this tutorial, we’ll build a simple question-answering application over a text data source.

The image below illustrates the provided user instruction and model answer examples.

![example](https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/87770915-1742-43d7-903b-b5960eda8011)

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