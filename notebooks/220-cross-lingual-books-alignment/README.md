# Cross-lingual Books Alignment With Transformers and OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F220-cross-lingual-books-alignment%2F220-cross-lingual-books-alignment.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/220-cross-lingual-books-alignment/220-cross-lingual-books-alignment.ipynb)

Cross-lingual text alignment is the task of matching sentences in a pair of texts that are translations of each other. In this notebook, you'll learn how to use a deep learning model to create a parallel book in English and German

This method helps you learn languages but also provides parallel texts that can be used to train machine translation models. This is particularly useful if one of the languages is low-resource or you don't have enough data to train a full-fledged translation model.

The notebook shows how to accelerate the most computationally expensive part of the pipeline - getting vectors from sentences - using the OpenVINO™ framework.

Pipeline
The notebook guides you through the entire process of creating a parallel book: from obtaining raw texts to building a visualization of aligned sentences. Here is the pipeline diagram:

<img src="https://user-images.githubusercontent.com/51917466/254582697-18f3ab38-e264-4b2c-a088-8e54b855c1b2.png"/>

By visualizing the result, you can evaluate which steps in the pipeline can be improved, which is also indicated in the diagram.

## Visualization

<img src="https://user-images.githubusercontent.com/51917466/254583328-7e5cf756-fa4c-4427-84be-04326433cd9a.png">

<img src="https://user-images.githubusercontent.com/51917466/254583163-3bb85143-627b-4f02-b628-7bef37823520.png">

## Benchmark
<img src="https://user-images.githubusercontent.com/51917466/254592797-0dbb6eb5-4035-48ae-b16f-5e09b93a7e72.png">
