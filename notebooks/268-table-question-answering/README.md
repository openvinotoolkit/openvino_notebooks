# Table Question Answering using TAPAS and OpenVINO™

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/268-table-question-answering/268-table-question-answering.ipynb)

Table Question Answering (Table QA) is the answering a question about an information on a given table. You can use the 
Table Question Answering models to simulate SQL execution by inputting a table.

In this tutorial we demonstrate how to use [the base example](https://huggingface.co/tasks/table-question-answering).
with OpenVINO. This example based on [TAPAS base model fine-tuned on WikiTable Questions (WTQ)](https://huggingface.co/google/tapas-base-finetuned-wtq) 
that is based on the paper [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://arxiv.org/abs/2004.02349#:~:text=Answering%20natural%20language%20questions%20over,denotations%20instead%20of%20logical%20forms).

Answering natural language questions over tables is usually seen as a semantic parsing task. To alleviate the 
collection cost of full logical forms, one popular approach focuses on weak supervision consisting of denotations 
instead of logical forms. However, training semantic parsers from weak supervision poses difficulties, and in addition, 
the generated logical forms are only used as an intermediate step prior to retrieving the denotation. 
In [this paper](https://arxiv.org/pdf/2004.02349.pdf), it is presented TAPAS, an approach to question answering over 
tables without generating logical forms. TAPAS trains from weak supervision, and predicts the denotation by selecting 
table cells and optionally applying a corresponding aggregation operator to such selection. TAPAS extends BERT's 
architecture to encode tables as input, initializes from an effective joint pre-training of text segments and tables 
crawled from Wikipedia, and is trained end-to-end.

## Notebook contents
The tutorial consists from following steps:

- Prerequisites
- Use the original model to run an inference
- Convert the original model to OpenVINO Intermediate Representation (IR) format
- Run the OpenVINO model
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).