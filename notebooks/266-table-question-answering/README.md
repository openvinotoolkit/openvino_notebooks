# TAPAS base model fine-tuned on WikiTable Questions (WTQ) with OpenVINO

Table Question Answering (Table QA) is the answering a question about an information on a given table. You can use the 
Table Question Answering models to simulate SQL execution by inputting a table.

Answering natural language questions over tables is usually seen as a semantic parsing task. To alleviate the collection
 cost of full logical forms, one popular approach focuses on weak supervision consisting of denotations instead of 
 logical forms. However, training semantic parsers from weak supervision poses difficulties, and in addition, the 
 generated logical forms are only used as an intermediate step prior to retrieving the denotation. In this paper, 
 we present TAPAS, an approach to question answering over tables without generating logical forms. TAPAS trains 
 from weak supervision, and predicts the denotation by selecting table cells and optionally applying a corresponding 
 aggregation operator to such selection. TAPAS extends BERT's architecture to encode tables as input, initializes from 
 an effective joint pre-training of text segments and tables crawled from Wikipedia, and is trained end-to-end. We 
 experiment with three different semantic parsing datasets, and find that TAPAS outperforms or rivals semantic parsing 
 models by improving state-of-the-art accuracy on SQA from 55.1 to 67.2 and performing on par with the state-of-the-art 
 on WIKISQL and WIKITQ, but with a simpler model architecture. We additionally find that transfer learning, which is 
 trivial in our setting, from WIKISQL to WIKITQ, yields 48.7 accuracy, 4.2 points above the state-of-the-art.

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