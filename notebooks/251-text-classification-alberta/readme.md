# Alberta Text Classification and Optimization with OpenVINO
This project demonstrates how to perform text classification using the Alberta model with OpenVINO, a framework that optimizes and accelerates machine learning models. By using OpenVINO, we can improve the performance of our text classification model and make it more efficient for deployment on different platforms.

The Alberta model is a pre-trained language model that has been fine-tuned on the Microsoft Research Paraphrase Corpus (MRPC) dataset. It can classify whether an input sentence is paraphrased or not. In this project, we will be using the Alberta model to classify text.

Getting Started
Before running the code, make sure you have the following libraries installed:

OpenVINO
PyTorch
Transformers
Once the libraries are installed, you can download the Alberta model from the Hugging Face model hub. The model is available at [Alberta](https://huggingface.co/textattack/albert-base-v2-MRPC?text=I+like+you.+I+love+you).

Running the Code
The project consists of three main parts:

Converting the PyTorch model to an OpenVINO IR format.
Running inference on the converted model.
Optimizing using NNCF (INT-8 quantization)


Conclusion
This project demonstrated how to perform text classification and optimization using the Alberta model with OpenVINO. By using OpenVINO, we were able to optimize and accelerate the model, making it more efficient for deployment on different platforms. With this knowledge, you can apply OpenVINO to other machine learning models to improve their performance and efficiency.