# Fill Mask with Bert-Base-Uncased
Masked language modeling is the task of masking some of the words in a sentence and predicting which words should replace those masks. These models are useful when we want to get a statistical understanding of the language in which the model is trained in.
![image](https://user-images.githubusercontent.com/95271966/206129198-524456ff-8fd3-4a61-82ed-cd8322cddfc3.png)
![image](https://user-images.githubusercontent.com/95271966/206129237-eb983579-a58e-4897-a307-af55d8f76158.png)

## Notebook Contents
This notebook performs fill mask operation using OpenVINO API 2.0. Fill mask is basically the method to predict the next word of a sentence based on the previous word. Model supports exactly one word to be predicted anywhere in the sentence. The predicted word is hidden as ['MASK'] tag. If no ['MASK'] tag is provided in the sentence, the model automatically adds a ['MASK'] tag at the end of the sentence. We'll be using bert-base-uncased transformers based model from Huggingface. Details of the model can be found [here].(https://huggingface.co/bert-base-uncased)

## Installation Instructions

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the packages required to run this project.
