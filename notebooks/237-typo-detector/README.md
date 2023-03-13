# Convert and Optimize Typo  Detector with OpenVINO™

Typo detection in AI is a process of identifying and correcting typographical errors in text data using machine learning algorithms. The goal of typo detection is to improve the accuracy, readability, and usability of text by identifying and correcting mistakes made during the writing process.

A typo detector takes a sentence as an input and identify all typographical errors such as misspellings and homophone errors.

This tutorial provides how to use the [Typo Detector](https://huggingface.co/m3hrdadfi/typo-detector-distilbert-en) from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library to perform the above task.

The model has been pretrained on the [NeuSpell](https://github.com/neuspell/neuspell) dataset.

<img src=https://user-images.githubusercontent.com/80534358/224564463-ee686386-f846-4b2b-91af-7163586014b7.png>

## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run and optimize the model.

The tutorial consists of the following steps:
- Required libraries
- Two possible methods
- Use the HuggingFace Optimum library
- Using the pipeline to use the model
- Conversion to ONNX format
- Conversion to OpenVINO IR format
- Run inference
- Required helper functions

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
