# RAG Performance & Fairness Evaluation Toolkit (OpenVINO + LangChain)

This toolkit enables developers to build, evaluate, and optimize Retrieval-Augmented Generation (RAG) applications with comprehensive quality metrics including accuracy, bias detection, and perplexity analysis plus a racial-bias indicator. This uses RAG pipeline optimized with Intel OpenVINO for enhanced performance on CPU, GPU, and NPU.  The pipeline leverages:
- Optimum-Intel’s `OVModelForCausalLM` with the OpenVINO backend for efficient inference.
- LangChain for orchestration of document loading, chunking, embedding, retrieval, reranking, and generation.

> Goal: Provide a portable notebook-driven workflow for rapid experimentation, model comparison, and validation of RAG systems on custom/private corpora.

---

## 1. What Is RAG?

Retrieval-Augmented Generation combines:
1. Retrieval: Selecting the most relevant context snippets from a document store.
2. Generation: Supplying those snippets to an LLM to produce grounded answers.

Benefits:
- Injects up-to-date and domain-specific knowledge without fine-tuning the LLM.
- Reduces hallucinations by constraining generation to retrieved evidence.
- Supports compliance and audit by exposing sources (metadata) for each answer.

---

## 2. RAG Performance & Fairness Evaluation Toolkit Overview

| Component                | Role |
|--------------------------|------|
| Document Loaders         | Ingest local files (.pdf, .txt, .docx, .json, .csv) or URLs/web pages. |
| Text Splitter            | Chunk documents into semantically sized pieces for embedding. |
| Embedding Model          | Converts chunks to vector representations for similarity search. |
| Vector Store / Index     | Persists embeddings enabling fast approximate or exact nearest-neighbor retrieval. |
| (Optional) Reranker      | Re-orders retrieved candidates for improved answer grounding. |
| Generator (OVModel)      | Runs local accelerated LLM inference via OpenVINO. |
| Evaluator                | Computes quality and bias metrics. |
| Notebook Orchestrator    | Step-by-step cells show the entire flow and allow interactive parameter tuning. |

---

## 3. Key Features

- **OpenVINO Model Optimization**: 
   - Hardware-accelerated inference using OpenVINO for LLMs and embedding models
- **Flexible Model Support**: 
  - LLM: Microsoft Phi-3-mini-4k-instruct (easily swappable with other HuggingFace models)
  - Embeddings: BGE-small-en-v1.5 (supports other embedding models)
  - Evaluation: Llama-2-7B for perplexity scoring
- **Advanced Retrieval**: 
  - ChromaDB vector store with persistent storage
  - FlashRank reranking for improved retrieval accuracy
  - Batch embedding insertion for large document sets
- **Multiple Document Sources**:
  - Web scraping from sitemaps and URLs
  - Local file loading (.pdf, .txt, .docx, .csv, .json, .xlsx)
  - Supports both single and bulk document processing
- **Comprehensive Evaluation Metrics**:
  - BLEU Score: Translation quality metric
  - ROUGE Score: Summary quality assessment
  - BERT Score: Semantic similarity using BERT embeddings
  - Perplexity: Language model confidence measurement
  - Diversity Score: Response variety analysis
  - Racial Bias Detection: Using hate-speech detection model

---

## 4. Installation

```bash
# Clone the repository
cd RAG-OV-Langchain
pip install -r requirements.txt
```

(If OpenVINO runtime prerequisites are not already satisfied, follow Intel’s OpenVINO setup instructions.)

---

## 5. Running the Notebook

1. Launch Jupyter: `jupyter notebook`
2. Open the provided notebook - `ov_rag_evaluator.ipynb`
3. Execute cells in order; each cell includes explanatory comments.
4. Provide input sources (file paths or URLs) when prompted.
5. Adjust parameters such as:
   - Chunk size / overlap
   - Embedding model name
   - Retrieval top-k
   - Reranker toggle
   - Generation temperature / max tokens
6. Run evaluation cells to view metrics dashboard output.

---

## 6. Input / Output Formats

### Supported Input
- Textual documents: `.pdf`, `.txt`, `.docx`, `.json`, `.csv`
- Web content: Page URLs (scraped & cleaned)
- (Extendable) Additional loaders can be registered for other data types.

### Output
- Generated answer grounded in retrieved context.
- List of source chunks with:
  - Document identifier
  - Chunk index
  - Similarity / relevance score
  - Optional rerank score
- Metrics report (per query or aggregate).

---

## 7. Evaluation Metrics

| Metric        | Purpose |
|---------------|---------|
| BERTScore     | Semantic similarity vs. reference answer(s). |
| BLEU          | n-gram precision (machine translation heritage; still indicative for overlap). |
| ROUGE         | Recall-oriented overlap (useful for summarization-style references). |
| Perplexity    | Fluency measure of generated text under a language model. |
| Racial Bias Indicator | Heuristic or embedding-based measure identifying disproportionate associations or skewed outputs. |

Notes:
- Provide one or more reference answers (gold annotations) for BLEU/ROUGE/BERTScore.
- Perplexity may rely on a reference language model distinct from the generator.
- Bias indicator may leverage word association tests or sentiment differentials; interpret conservatively.

---

## 8. Racial Bias Indicator (Concept)

The notebook computes a racial bias signal that can highlight when generated answers:
- Over-index on certain demographic terms.
- Exhibit asymmetric sentiment or descriptors.
- Associate professions or attributes disproportionately.

Recommended usage:
- Treat as a screening heuristic.
- Follow up with manual review.
- Do not treat a single numeric score as definitive.

---

## 9. Customization

You can modify:
- Embedding backend (e.g., `sentence-transformers`, `text-embedding-*` models).
- Retrieval strategy (FAISS, chroma, or other vector stores).
- Reranking (e.g., cross-encoder or LLM-based rerank).
- Generation model (swap Hugging Face model; ensure OpenVINO export or optimization).
- Metric thresholds for acceptance gating.

---

## 10. Suggested Workflow

1. Curate domain corpus.
2. Run baseline RAG with default parameters.
3. Collect queries & gold references (if available).
4. Evaluate metrics; record baseline.
5. Iterate:
   - Tune chunking, top-k.
   - Introduce reranker.
   - Switch embedding model.
   - Optimize LLM (quantization, OpenVINO optimizations).
6. Compare metric deltas; choose best configuration for deployment.

---

## 11. Performance Considerations

- OpenVINO accelerates inference on Intel hardware (CPU / GPU / NPU where supported).
- Smaller embedding models may trade slight recall for speed.
- Reranking adds latency; enable only if precision gains matter.
- Batch queries in evaluation phase to amortize setup costs.

---

## 12. Limitations

- Metrics may not fully capture factual grounding; consider human review.
- Bias indicator is heuristic; deeper audits require specialized tools.
- Long documents may need advanced chunking strategies (semantic splitting).
- URL ingestion quality depends on HTML cleanliness.

---

## FAQs

Q: Can I use a different LLM?  
A: Yes, replace the checkpoint and ensure OpenVINO optimization/export steps are applied.

Q: Do I need gold answers?  
A: For BLEU/ROUGE/BERTScore, yes. For exploratory retrieval quality, you can still inspect sources without them.

Q: How to reduce hallucinations?  
A: Increase retrieval relevance (tune embeddings, use reranking) and constrain generation parameters (lower temperature).

---
