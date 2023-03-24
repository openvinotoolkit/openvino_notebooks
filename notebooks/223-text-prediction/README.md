# Text Prediction with OpenVINO
Text generation is a type of natural language processing that uses computational linguistics and artificial intelligence to automatically produce text that can meet specific communicative needs.


In this demo we have the models:

* **Generative Pre-trained Transformer 2 ([GPT-2](https://github.com/openai/gpt-2/blob/master/model_card.md))** model for text prediction.
* **Generative Pre-trained Transformer Neo ([GPT-Neo](https://github.com/EleutherAI/gpt-neo))** model for text prediction.
* **PersonaGPT ([PersonaGPT](https://huggingface.co/af1tang/personaGPT))** model for Conversation
The complete pipeline of this Text Generation is shown below.

![image2](https://user-images.githubusercontent.com/91228207/163990722-d2713ede-921e-4594-8b00-8b5c1a4d73b5.jpeg)

This is a demonstration in which the user can type the beginning of the text and the network will generate a further. This procedure can be repeated as many times as the user desires.

The following images show an example of the input sequence and corresponding predicted sequence.

* GPT-2:
![image](https://user-images.githubusercontent.com/91228207/185103977-54b1671a-f02c-4f4b-9722-5c4e8b119fc7.png)
* GPT-Neo:
![image](https://user-images.githubusercontent.com/95569637/223999855-32c15531-0f41-42ee-a318-0f5b5ebd687e.png)

The Modified Pipeline For Conversation is shown below.

![image2](https://user-images.githubusercontent.com/95569637/226101538-e204aebd-a34f-4c8b-b90c-5363ba41c080.jpeg)

This is a demonstration in which a user can have a conversation with the model. The user provides some input based on which the model generates a response. The user can further talk to the model as if having a chat with the model.

The following image shows an example of a conversation.

![image](https://user-images.githubusercontent.com/95569637/226102963-ab545346-175b-4eb5-94f6-f2c0a4d573f5.png)
## Notebook Contents

The notebook demonstrates text prediction with OpenVINO using the following models

* [gpt-2](https://huggingface.co/gpt2) model from HuggingFace Transformers.
* [gpt-neo](https://huggingface.co/EleutherAI/gpt-neo-125M) model from HuggingFace Transformers.
* [personaGPT](https://huggingface.co/af1tang/personaGPT) model from HuggingFace Transformers
## Installation Instructions

If you have not done so already, please follow the [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md) to install all required dependencies.
