# Document Visual Question Answering Using Pix2Struct and OpenVINO
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/260-pix2struct-docvqa/260-pix2struct-docvqa.ipynb)

DocVQA (Document Visual Question Answering) is a research field in computer vision and natural language processing that focuses on developing algorithms to answer questions related to the content of a document represented in image format, like a scanned document, screenshots, or an image of a text document. Unlike other types of visual question answering, where the focus is on answering questions related to images or videos, DocVQA is focused on understanding and answering questions based on the text and layout of a document. The questions can be about any aspect of the document text. DocVQA requires understanding the document’s visual content and the ability to read and comprehend the text in it.

DocVQA offers several benefits compared to OCR (Optical Character Recognition) technology:
* Firstly, DocVQA can not only recognize and extract text from a document, but it can also understand the context in which the text appears. This means it can answer questions about the document’s content rather than simply provide a digital version.
* Secondly, DocVQA can handle documents with complex layouts and structures, like tables and diagrams, which can be challenging for traditional OCR systems.
* Finally, DocVQA can automate many document-based workflows, like document routing and approval processes, to make employees focus on more meaningful work. The potential applications of DocVQA include automating tasks like information retrieval, document analysis, and document summarization.

[Pix2Struct](https://arxiv.org/pdf/2210.03347.pdf) is a multimodal model for understanding visually-situated language that easily copes with extracting information from images. The model is trained using the novel learning technique to parse masked screenshots of web pages into simplified HTML, providing a significantly well-suited pretraining data source for the range of downstream activities such as OCR, visual question answering and image captioning.

In this tutorial we consider how to run Pix2Struct model using OpenVINO for solving document visual question answering task. We will use pre-trained model from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library. To simplify the user experience, the [Hugging Face Optimum](https://huggingface.co/docs/optimum) library is used to convert the model to OpenVINO™ IR format.

## Notebook Contents

The tutorial consist of the following steps:

- Install prerequisites
- Download and convert model from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino).
- Test model inference
- Launch interactive demo

As the result, will be created inference pipeline which accepts image of the document and question about it and generates answer for given question.
The result of running demonstration shown bellow:

![](https://user-images.githubusercontent.com/29454499/276283074-df7464e6-8293-4c6c-8f77-8e95d8f94c11.png)



## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

