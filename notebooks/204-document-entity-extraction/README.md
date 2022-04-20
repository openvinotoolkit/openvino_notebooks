# Document Entity Extraction with OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F204-document-entity-extraction%2F204-document-entity-extraction.ipynb)

Simple Text            | Image File          |  PDF File
:-------------------------:|:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/33627846/159575177-bbcd5053-a6c6-4c0c-a1e3-b1c9f0f1e319.png)  |  ![](https://user-images.githubusercontent.com/33627846/159575191-646cc9a6-4588-4064-9717-6ae0be7bd982.png)  |  ![](https://user-images.githubusercontent.com/33627846/159575210-092c1822-79d7-40fe-ac3b-55ba7b7afa15.png)


Document Entity extraction is a part of information retrieval and natural language processing (NLP), which is able to extract entities. Such NER (Named Entity Recognition) systems can pull entities from an unstructured collection of natural language documents called, in that case, knowledge base. This unstructured data can be a simple text, image or even a PDF file. NER from an image or PDF file also includes an optical character recognition (OCR) preprocessing step before launching entity extraction.

## Notebook Contents

This notebook demonstrates entity extraction with OpenVINO using the [Squad-tuned BERT](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-small-uncased-whole-word-masking-squad-int8-0002) model from Open Model Zoo.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.