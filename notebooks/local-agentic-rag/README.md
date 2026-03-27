# 🤖 Local RAG Pipeline with Ollama and Optional Agentic Workflow

This notebook demonstrates a **minimal, fully local Retrieval-Augmented Generation (RAG) pipeline** using Ollama, ChromaDB, and an optional agentic workflow with LangGraph.

The implementation is designed to be **educational, modular, and CPU-friendly**, requiring no cloud APIs after initial setup.

---

## 📚 Overview

This notebook walks through building a complete local AI pipeline:

- Local LLM inference using Ollama
- Document embedding and storage with ChromaDB
- Retrieval-Augmented Generation (RAG)
- Optional agentic workflow using LangGraph
- Optional OpenVINO™ integration for optimized inference

The goal is to provide a **clear and reproducible introduction** to local-first AI systems.

---

## 🔍 What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances LLM responses by retrieving relevant context from a knowledge base before generating an answer.

This helps:
- Reduce hallucinations
- Incorporate domain-specific knowledge
- Improve factual accuracy

---

## 🤖 Optional Agentic Workflow

This notebook includes an **optional agentic extension** using LangGraph.

In this setup, the system can:
- Decide whether retrieval is needed
- Route queries dynamically
- Use simple tools such as a calculator

> ⚠️ This section is optional and intended for learning purposes.  
> The core RAG pipeline works independently without the agentic extension.

---

## ⚡ OpenVINO™ Integration

OpenVINO™ is Intel’s toolkit for optimizing and deploying deep learning models.

This notebook is designed to be **compatible with OpenVINO optimization workflows**, including:

- Model conversion (FP32 → FP16 / INT8)
- Quantization and compression
- CPU, GPU, and NPU performance optimization

> 💡 OpenVINO integration is optional. The notebook can run without it.

---

## 💻 Requirements

| Component | Requirement |
|----------|-------------|
| Python | 3.9+ |
| RAM | 8 GB minimum, 16 GB recommended |
| Storage | ~5 GB free |
| OS | Windows, Linux, or macOS |

> ✅ No GPU is required.

---

## 🛠️ Setup Instructions

### 1. Install Python dependencies

```bash
pip install ollama chromadb langgraph langchain sentence-transformers jupyter