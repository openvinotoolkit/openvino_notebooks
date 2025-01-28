# Unified image generation using OmniGen and OpenVINO

OmniGen is a unified image generation model that can generate a wide range of images from multi-modal prompts. It is designed to be simple, flexible, and easy to use. Existing image generation models often require loading several additional network modules (such as ControlNet, IP-Adapter, Reference-Net, etc.) and performing extra preprocessing steps (e.g., face detection, pose estimation, cropping, etc.) to generate a satisfactory image. OmniGen can generate various images directly through arbitrarily multi-modal instructions without additional plugins and operations.  it can automatically identify the features (e.g., required object, human pose, depth mapping) in input images according to the text prompt.

Here are the illustrations of OmniGen's capabilities:

* You can control the image generation flexibly via OmniGen


  ![exanple_1.png](https://github.com/VectorSpaceLab/OmniGen/raw/main/imgs/demo_cases.png)
  
* Referring Expression Generation: You can input multiple images and use simple, general language to refer to the objects within those images. OmniGen can automatically recognize the necessary objects in each image and generate new images based on them. No additional operations, such as image cropping or face detection, are required.

  ![example_2.png](https://github.com/VectorSpaceLab/OmniGen/raw/main/imgs/referring.png)


You can find more details about a model on [project page](https://vectorspacelab.github.io/OmniGen/), [paper](https://arxiv.org/pdf/2409.11340v1), [original repository](https://github.com/VectorSpaceLab/OmniGen).

This tutorial considers how to run and optimize OmniGen using OpenVINO.

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model using OpenVINO and NNCF
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll try to run OmniGen for various image generation tasks.

The image bellow is illustrates model's result for text to image generation.
![](https://github.com/user-attachments/assets/ca0929af-f766-4e69-872f-95ceceeac634)


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/omnigen/README.md" />
