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
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-rag-langchain/README.md" />

## Evaluating the RAG Pipeline
The python notebook (`llm-rag-langchain-eval.ipynb`) enables developers to build, evaluate, and optimize Retrieval-Augmented Generation (RAG) applications with comprehensive quality metrics including accuracy, bias detection, and perplexity analysis plus a racial-bias indicator. This uses RAG pipeline optimized with Intel OpenVINO for enhanced performance on CPU, GPU, and NPU. The pipeline leverages:

Optimum-Intel’s OVModelForCausalLM with the OpenVINO backend for efficient inference.
LangChain for orchestration of document loading, chunking, embedding, retrieval, reranking, and generation.

To run the notebook:

1. Launch Jupyter: `jupyter notebook`
2. Open the provided notebook — `llm-rag-langchain-eval.ipynb`
3. Execute cells in order; each cell includes explanatory comments.
4. Provide input sources (file paths or URLs) when prompted.
5. Adjust parameters such as:
   - Chunk size / overlap
   - Embedding model name
   - Retrieval top-k
   - Reranker toggle
   - Generation temperature / max tokens
6. Run evaluation cells to view metrics dashboard output.

Supported Input
  Textual documents: `.pdf`, `.txt`, `.docx`, `.json`, `.csv`
  Web content: Page URLs (scraped & cleaned)
  (Extendable) Additional loaders can be registered for other data types.
Output
  Generated answer grounded in retrieved context.
  List of source chunks with:
    Document identifier
    Chunk index
    Similarity / relevance score
    Optional rerank score
    Metrics report (per query or aggregate).

Evaluation Metrics

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
