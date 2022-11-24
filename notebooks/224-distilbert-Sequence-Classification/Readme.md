# Sequence Classification With OpenVINO API 2.0
This notebook performs text classification using OpenVINO API 2.0 using `distilbert-base-uncased` model from HuggingFace transformers finetuned on [SST-2](https://huggingface.co/datasets/sst2) dataset. We'll convert the model to onnx using the transformers.onnx package from the transformers library and perform inference using OpenVINO. More information about this model can be found in [model card](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english). You can alternatively replace the model with the other bert-based models for sequence classification.


Output from a user input would look like this.

![image](https://user-images.githubusercontent.com/95271966/203713415-669a755d-3243-4e66-b19b-cd17774a1a64.png)


Output from a file would look something like this.
![image](https://user-images.githubusercontent.com/95271966/203713154-b78b383e-ec42-4b3b-a142-47b00640bdea.png)


## Notebook Contents
We'll be using distilbert-base-uncased-finetuned-sst-2-english transformers based model from Huggingface. Details of the model can be found [here].(https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

## Installation Instructions

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the packages required to run this project.

```bash
pip install -r requirements.txt
```

