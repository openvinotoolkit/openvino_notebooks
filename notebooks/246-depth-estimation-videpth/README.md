# Depth estimation using VI-depth with OpenVINO™

<p align="center" width="100%">
    <img src="https://raw.githubusercontent.com/alexklwong/void-dataset/master/figures/void_samples.png"> 
    <figcaption>
        <span class="caption"> <i> A couple of depth samples taken during the course of data collection.</i> </span>
        <i class="photo-credit"> Figure from Alex Wong's void dataset.</i>
    </figcaption>
</p>

A visual-inertial depth estimation pipeline that integrates monocular depth estimation and visual-inertial odometry to produce dense depth estimates with metric scale has been demonstrated via this notebook. 

The entirety of this notebook tutorial has been adapted from the [VI-Depth repository](https://github.com/isl-org/VI-Depth). Some pieces of the code dealing with the inference have been adapted as it is the [utils](vi_depth_utils) directory and have been used in the notebook. The data for inference has been obtained as a subset of the data that has been linked [here](https://github.com/alexklwong/void-dataset/blob/master/README.md). Due to the **compressed** format of the data in openly available Google drive links, uncompressing the same for few inference examples is not recommended. Hence this OpenVINO™ tutorial downloads data *on the fly*.

The authors have published their work here:

> [Monocular Visual-Inertial Depth Estimation](https://arxiv.org/abs/2303.12134)  
> Diana Wofk, René Ranftl, Matthias Müller, Vladlen Koltun


## Notebook contents

The notebook contains a detailed tutorial of the visual-inertial depth estimation pipeline as follows:
1. Import required packages and install the PyTorch deep learning library associated with image models (`timm v.0.6.12`).
2. Download an appropriate depth predictor from the [VI-Depth repository](https://github.com/isl-org/VI-Depth/).
3. Get the appropriate predictor model corresponding to the above _depth_ (a `PyTorch` model callable).
4. Download images and their depth maps for dummy input creation.
5. Convert and transform the models to the [OpenVINO IR](https://docs.openvino.ai/latest/openvino_ir.html) representation for inference using the dummy inputs from 4.
6. Download another set of images similar to the one used in step 4. and finally run the models against them.


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](https://github.com/pronoym99/openvino_notebooks/blob/main/README.md#-installation-guide).
