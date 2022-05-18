# Document Entity Extraction with OpenVINO

![](https://user-images.githubusercontent.com/33627846/169033561-2b10abc0-abb6-4d10-afbe-da921ed77432.gif)

Document Entity extraction is a part of information retrieval and natural language processing (NLP), which is able to extract entities. Given a group of text or documents of any format, recognizing and extracting specific types of entities is called Named Entity Extraction (NER). It's used across industries in use cases related to data enrichment, content recommendation, customer support, advanced search algorithms etc.

NER systems can pull entities from an unstructured collection of natural language documents called, in that case, knowledge base. This unstructured data can be a simple text, image or even a PDF file. NER from an image or PDF file also includes an optical character recognition (OCR) preprocessing step before launching entity extraction. This notebook is continuation of [213-question-answering](https://github.com/openvinotoolkit/openvino_notebooks/tree/204-nlp-document-inference/notebooks/213-question-answering) notebook.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F204-document-entity-extraction%2F204-document-entity-extraction.ipynb)<br> *Binder is a free service where the webcam will not work, and performance on the video will not be good. For best performance, we recommend installing the notebooks locally.*

## Notebook Contents

This notebook demonstrates entity extraction from either a simple text, an image or a PDF file, with OpenVINO using the [Squad-tuned BERT](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-small-uncased-whole-word-masking-squad-int8-0002) model from Open Model Zoo. In the notebook we show how to create the following pipeline:

<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/33627846/166122112-d5c45a4c-892e-438c-aff2-003368bdcad5.png"> 
</p>


## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.