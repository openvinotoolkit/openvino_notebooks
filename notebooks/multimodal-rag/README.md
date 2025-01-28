# Multimodal RAG for video analytics with LlamaIndex

Constructing a RAG pipeline for text is relatively straightforward, thanks to the tools developed for parsing, indexing, and retrieving text data. However, adapting RAG models for video content presents a greater challenge. Videos combine visual, auditory, and textual elements, requiring more processing power and sophisticated video pipelines.

To build a truly multimodal search for videos, you need to work with different modalities of a video like spoken content, visual. In this notebook, we showcase a Multimodal RAG pipeline designed for video analytics. It utilizes Whisper model to convert spoken content to text, CLIP model to generate multimodal embeddings, and Vision Language model (VLM) to process retrieved images and text messages. The following picture illustrates how this pipeline is working.

![image](https://github.com/user-attachments/assets/a8ebf3fc-7a34-416b-b744-609965792744)

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Download and process video
- Create the multi-modal index
- Search text and image embeddings
- Generate final response using VLM
- Launch Interactive demo

In this demonstration, you'll create interactive Q&A system that can answer questions about provided video's content.

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/multimodal-rag/README.md" />
