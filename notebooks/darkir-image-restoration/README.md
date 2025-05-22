# Low-Light Image Restoration with DarkIR model using OpenVINO™

[DarkIR](https://arxiv.org/abs/2412.13443) is an CNN model for Low-Light Image Restoration.

Photography during night or in dark conditions typically suffers from noise, low light and blurring issues due to the dim environment and the common use of long exposure. Although Deblurring and Low-light Image Enhancement (LLIE) are related under these conditions, most approaches in image restoration solve these tasks separately. DarkIR is an efficient and robust neural network for multi-task low-light image restoration. This approach proposes new attention mechanisms to enhance the receptive field of efficient CNNs. Current method reduces the computational costs in terms of parameters and MAC operations compared to previous methods. DarkIR achieves new state-of-the-art results on the popular LOLBlur, LOLv2 and Real-LOLBlur datasets, being able to generalize on real-world night and dark images.

![networks-scheme](https://raw.githubusercontent.com/cidautai/DarkIR/refs/heads/main/assets/networks-scheme.png)

[**image_source*](https://github.com/cidautai/DarkIR/blob/main/assets/networks-scheme.png)

You can find more information about this model in the [research paper](https://arxiv.org/abs/2412.13443) and GitHub [repository](https://github.com/cidautai/DarkIR).


The notebook contains the following steps:

1. Prepare the PyTorch model.
2. Prepare data and perform inference of the PyTorch model with image.
3. Convert model to OpenVINO IR, using the model conversion API.
4. Run inference with OpenVINO with image and video.
5. Launch interactive demo


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md)..

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/darkir-image-restoration/README.md" />
