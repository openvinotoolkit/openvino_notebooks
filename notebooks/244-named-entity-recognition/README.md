# Named entity recognition with OpenVINO™
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/244-named-entity-recognition/244-named-entity-recognition.ipynb)

Named entity recognition (NER) is one of the most popular data processing tasks. It is a natural language processing (NLP) method that involves detecting key information in unstructured text and categorizing it into pre-defined categories. These categories or named entities refer to the key subjects of the text, such as names, locations, companies and etc.
NER is a suitable method for situations when a high-level overview of a large amount of text is needed. NER can be helpful with such tasks as analyzing key information in unstructured text or automating the information extraction of large amounts of data.

This tutorial shows how to perform named entity recognition using OpenVINO. We will use the pre-trained model [`elastic/distilbert-base-cased-finetuned-conll03-english`](https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english). It is DistilBERT based model, trained on [`conll03 english dataset`](https://huggingface.co/datasets/conll2003). The model can recognize four named entities in text: persons, locations, organizations, and names of miscellaneous entities that do not belong to the previous three groups. The model is sensitive to capital letters.

To simplify the user experience, the [Hugging Face Optimum](https://huggingface.co/docs/optimum) library is used to convert the model to OpenVINO™ IR format and quantize it.

## Notebook Contents

The tutorial consists of the following steps:

* Download the model
* Quantize and save the model in OpenVINO IR format
* Prepare demo for Named Entity Recognition OpenVINO Runtime
* Compare the Original and Quantized Models


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

### See Also

* [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
* [Hugging Face Optimum](https://huggingface.co/docs/optimum)
