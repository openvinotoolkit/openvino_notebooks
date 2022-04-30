# Document Entity Extraction with OpenVINO

![](https://user-images.githubusercontent.com/33627846/164333009-b6ee55c9-ad9a-4d90-822a-f05069537a13.gif)

Document Entity extraction is a part of information retrieval and natural language processing (NLP), which is able to extract entities. Such NER (Named Entity Recognition) systems can pull entities from an unstructured collection of natural language documents called, in that case, knowledge base. This unstructured data can be a simple text, image or even a PDF file. NER from an image or PDF file also includes an optical character recognition (OCR) preprocessing step before launching entity extraction.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F204-document-entity-extraction%2F204-document-entity-extraction.ipynb)<br> *Binder is a free service where the webcam will not work, and performance on the video will not be good. For best performance, we recommend installing the notebooks locally.*

## Notebook Contents

This notebook demonstrates entity extraction with OpenVINO using the [Squad-tuned BERT](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-small-uncased-whole-word-masking-squad-int8-0002) model from Open Model Zoo.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.