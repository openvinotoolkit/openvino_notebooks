# Florence-2: Open Source Vision Foundation Model

Florence-2 is a lightweight vision-language foundation model developed by Microsoft Azure AI and open-sourced under the MIT license. It aims to achieve a unified, prompt-based representation for diverse vision and vision-language tasks, including captioning, object detection, grounding, and segmentation. Despite its compact size, Florence-2 rivals much larger models like [Kosmos-2](../kosmos2-multimodal-large-language-model/kosmos2-multimodal-large-language-model.ipynb) in performance. Florence-2 represents a significant advancement in vision-language models by combining lightweight architecture with robust capabilities, making it highly accessible and versatile. Its unified representation approach, supported by the extensive FLD-5B dataset, enables it to excel in multiple vision tasks without the need for separate models. This efficiency makes Florence-2 a strong contender for real-world applications, particularly on devices with limited resources.

More details about model can be found in [model's resources collection](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de) and original [paper](https://arxiv.org/abs/2311.06242).

In this tutorial we consider how to convert and run Florence2 using OpenVINO.

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll try to run model on various vision tasks including object detection, image captioning and text recognition.
![dogs.png](https://github.com/user-attachments/assets/b2469455-8ab6-4718-8fe0-3e9ea17ec1ce)


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/florence2/README.md" />