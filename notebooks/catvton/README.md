# Virtual Try-On with CatVTON and OpenVINO

### Abstract
Virtual try-on methods based on diffusion models achieve realistic try-on effects but replicate the backbone network as a ReferenceNet or leverage additional image encoders to process condition inputs, resulting in high training and inference costs. [In this work](http://arxiv.org/abs/2407.15886), authors rethink the necessity of ReferenceNet and image encoders and innovate the interaction between garment and person, proposing CatVTON, a simple and efficient virtual try-on diffusion model.
It facilitates the seamless transfer of in-shop or worn garments of arbitrary categories to target persons by simply
concatenating them in spatial dimensions as inputs. The efficiency of the model is demonstrated in three aspects: 
 1. Lightweight network. Only the original diffusion modules are used, without additional network modules. The text encoder and cross attentions for text injection in the backbone are removed, further reducing the parameters by 167.02M.
 2. Parameter-efficient training. We identified the try-on relevant modules through experiments and achieved high-quality try-on effects by training only 49.57M parameters (∼5.51% of the backbone network’s parameters). 
 3. Simplified inference. CatVTON eliminates all unnecessary conditions and preprocessing steps, including pose estimation, human parsing, and text input, requiring only garment reference, target person image, and mask for the virtual try-on process. Extensive experiments demonstrate that CatVTON achieves superior qualitative and quantitative results with fewer prerequisites and trainable parameters than baseline methods. Furthermore, CatVTON shows good generalization in in-the-wild scenarios despite using open-source datasets with only 73K samples.


Teaser image from [CatVTON GitHub](https://github.com/Zheng-Chong/CatVTON)
![teaser](https://github.com/Zheng-Chong/CatVTON/blob/edited/resource/img/teaser.jpg?raw=true)

In this tutorial we consider how to convert, optimize and run this model using OpenVINO. An additional part demonstrates how to run optimization with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up pipeline.

## Notebook contents
This tutorial consists of the following steps:
- Prerequisites
- Convert the model to OpenVINO IR
- Compiling models
- Optimizing the model using NNCF Post-Training Quantization API
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/catvton/README.md" />
