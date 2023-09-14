# Convert and Optimize Typo Detector with OpenVINO™

Typo detection in AI is a process of identifying and correcting typographical errors in text data using machine learning algorithms. The goal of typo detection is to improve the accuracy, readability, and usability of text by identifying and correcting mistakes made during the writing process.

A typo detector takes a sentence as an input and identify all typographical errors such as misspellings and homophone errors.

This tutorial provides how to use the [Typo Detector](https://huggingface.co/m3hrdadfi/typo-detector-distilbert-en) from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library to perform the above task.

The model has been pretrained on the [NeuSpell](https://github.com/neuspell/neuspell) dataset.

<img src=https://user-images.githubusercontent.com/80534358/224564463-ee686386-f846-4b2b-91af-7163586014b7.png>

</br>

The notebook provides two methods to run the inference of typo detector with OpenVINO runtime, so that you can experience both calling the API of Optimum with OpenVINO Runtime included, and loading models in other frameworks, converting them to OpenVINO IR format, and running inference with OpenVINO Runtime.

1. Use the [Hugging Face Optimum](https://huggingface.co/docs/optimum/index) library to load the compiled model in OpenVINO IR format. Then create a pipeline with the loaded model to run inference.

2. Load the model and convert to OpenVINO IR.
   The Pytorch model is converted to [OpenVINO IR format](https://docs.openvino.ai/latest/openvino_ir.html). This method provides much more insight to how to set up a pipeline from model loading to model converting, compiling and running inference with OpenVINO, so that you could conveniently use OpenVINO to optimize and accelerate inference for other deep-learning models.

The following table summarizes the major differences between the two methods

</br>

| Method 1                                                   | Method 2                                                        |
| ---------------------------------------------------------- | --------------------------------------------------------------- |
| Load models from Optimum, an extension of transformers     | Load model from transformers                                    |
| Load the model in OpenVINO IR format on the fly            | Convert to OpenVINO IR                                          |
| Load the compiled model by default                         | Compile the OpenVINO IR and run inference with OpenVINO Runtime |
| Pipeline is created to run inference with OpenVINO Runtime | Manually run inference.                                         |

</br>

## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run and optimize the model.

The tutorial consists of the following steps:

- Required libraries
- Two possible methods
  - Using the HuggingFace Optimum library
    - Loading the compiled model in OpenVINO IR format
    - Creating the pipeline
    - Run inference/Demo
  - Using the pipeline to use the model
    - Conversion to OpenVINO IR format
    - Required helper functions
    - Run inference/Demo

</br>

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
