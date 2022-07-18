# Named Entity Recognition with OpenVINO™

![](https://user-images.githubusercontent.com/33627846/169470030-0370963e-6ad8-49e3-be7a-f02a2c677733.gif)

Named Entity Recognition (NER) is a part of information retrieval and natural language processing (NLP), which is able to extract entities. Given a group of sentences in form of text, recognizing and extracting specific types of entities is called Named Entity Extraction. It is used across industries in use cases related to data enrichment, content recommendation, customer support, text summarization, advanced search algorithms etc.

NER systems can pull entities from an unstructured collection of natural language documents called, in that case, knowledge base. This notebook  demonstrates entity recognition from unstructured data, using simple text, but this can be extended to images or even a PDF file. NER from an image or PDF file would also include an optical character recognition (OCR) preprocessing step before launching entity extraction. This notebook is a continuation of [213-question-answering](https://github.com/openvinotoolkit/openvino_notebooks/tree/204-nlp-document-inference/notebooks/213-question-answering) notebook.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F204-named-entity-recognition%2F204-named-entity-recognition.ipynb)<br> *Binder is a free service where the webcam will not work, and performance on the video will not be good. For the best performance, it is recommended to install the notebooks locally.*

## Notebook Contents

This notebook demonstrates entity extraction from a simple text, with OpenVINO, using the [Squad-tuned BERT](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-small-uncased-whole-word-masking-squad-int8-0002) model from Open Model Zoo. The notebook shows how to create the following simple pipeline:

<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/33627846/169458895-c418c7aa-9cec-41f0-b393-e9d756cf8fe4.png"> 
</p>


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).