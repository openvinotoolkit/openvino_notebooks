# Knowledge graphs model optimization using the Intel OpenVINO toolkit

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fknowledge-graphs-conve%2Fknowledge-graphs-conve.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/knowledge-graphs-conve/knowledge-graphs-conve.ipynb)

[Knowledge graphs](https://arxiv.org/pdf/2002.00388.pdf) provide an efficient way to represent domain information and are used in a number of real-world applications.
<br/><br/>
Commonly, knowledge graphs are used to encode domain information by representing them as factual triples of source, relation and target entities.
For example, for the countries domain, the domain knowledge can be encoded by representing the countries as entities (for example, Germany, Poland) and relation between them (Neighbor) as factual triples (Poland, Neighbor, Germany) as well as other information as follows:
<p style="text-align:center;">
    <img src="https://user-images.githubusercontent.com/29454499/210564312-82cda081-b2ed-4027-8251-ca57a97b4666.png" width=500/>
</p>
These representations can then be used for tasks such as entity prediction (predict the target entity given the source entity and relation) and link prediction (whether a factual link exists between two given entities). However, such entity and link prediction tasks on large ontologies or knowledge bases can be challenging and time consuming as they can contain millions of factual triples. Hence, several neural networks based embedding models have been proposed in recent literature for efficient representation of knowledge graphs.

## Notebook Contents

This notebook showcases how runtime (latency and throughput) performance of knowledge graph inference or prediction tasks can be further optimized on supported Intel® architectures by using the Intel® Distribution of OpenVINO™ Toolkit for one such knowledge graph embeddings model called ConvE. <br><br>
The ConvE knowledge graph embeddings model is an implementation of the paper: ["Convolutional 2D Knowledge Graph Embeddings"](https://arxiv.org/abs/1707.01476) by Tim Dettmers et al.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/knowledge-graphs-conve/README.md" />
