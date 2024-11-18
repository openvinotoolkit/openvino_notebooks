# Screen Parsing with OmniParser and OpenVINO

Recent breakthrough in Visual Language Processing and Large Language models made significant strides in understanding and interacting with the world through text and images. However, accurately parsing and understanding complex graphical user interfaces (GUIs) remains a significant challenge.
[OmniParser](https://microsoft.github.io/OmniParser/) is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements. This enables more accurate and efficient interaction with GUIs, empowering AI agents to perform tasks across various platforms and applications.

![](https://microsoft.github.io/OmniParser/static/images/flow_merged0.png)

More details about model can be found in [Microsoft blog post](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/), [paper](https://arxiv.org/pdf/2408.00203), [original repo](https://github.com/microsoft/OmniParser) and [model card](https://huggingface.co/microsoft/OmniParser). 

In this tutorial we consider how to run OmniParser using OpenVINO.

# Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll try to run OmniParser for recognition of UI elements on screenshots.


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/florence2/README.md" />
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/omniparser/README.md" />
