# Text Prediction with OpenVINO™
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/223-text-prediction/223-text-prediction.ipynb)

Text generation is a type of natural language processing that uses computational linguistics and artificial intelligence to automatically produce text that can meet specific communicative needs.


In this demo we have the models:

* **Generative Pre-trained Transformer 2 ([GPT-2](https://github.com/openai/gpt-2/blob/master/model_card.md))** model for text prediction.
* **Generative Pre-trained Transformer Neo ([GPT-Neo](https://github.com/EleutherAI/gpt-neo))** model for text prediction.
* **PersonaGPT ([PersonaGPT](https://huggingface.co/af1tang/personaGPT))** model for Conversation.

The complete pipeline of this Text Generation is shown below:


![image2](https://user-images.githubusercontent.com/91228207/163990722-d2713ede-921e-4594-8b00-8b5c1a4d73b5.jpeg)

This notebook demonstrates how to generate text using user input. By entering the beginning of a sentence or paragraph, the network generates additional text to complete the sequence. Also, This process can be repeated as many times as the user desires, and the model responds to each input, allowing users to engage in a conversation-like interaction.

The following images show an example of the input sequence and corresponding predicted sequence.

* GPT-2:
![image](https://user-images.githubusercontent.com/91228207/185103977-54b1671a-f02c-4f4b-9722-5c4e8b119fc7.png)
* GPT-Neo:
![image](https://user-images.githubusercontent.com/95569637/223999855-32c15531-0f41-42ee-a318-0f5b5ebd687e.png)

The Modified Pipeline For Conversation is shown below.

![image2](https://user-images.githubusercontent.com/95569637/226101538-e204aebd-a34f-4c8b-b90c-5363ba41c080.jpeg)

The following image shows an example of a conversation.

![image](https://user-images.githubusercontent.com/95569637/229706278-2aa6a60d-02f4-45e2-9541-97529df8359d.png)
## Notebook Contents

The notebook demonstrates text prediction with OpenVINO using the following models

* [GPT-2](https://huggingface.co/gpt2) model from HuggingFace Transformers.
* [GPT-Neo 125M](https://huggingface.co/EleutherAI/gpt-neo-125M) model from HuggingFace Transformers.
* [PersonaGPT](https://huggingface.co/af1tang/personaGPT) model from HuggingFace Transformers
## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
