# Depth estimation using VI-depth with OpenVINO™

<p align="center" width="100%">
    <img src="https://raw.githubusercontent.com/alexklwong/void-dataset/master/figures/void_samples.png">    
</p>

A visual-inertial depth estimation pipeline that integrates monocular depth estimation and visual-inertial odometry to produce dense depth estimates with metric scale has been demonstrated via this notebook. 

The entirety of this notebook tutorial has been adapted from this [repository](https://github.com/isl-org/VI-Depth). Some pieces of the code dealing with the inference have been adapted as it is the [utils](vi_depth_utils) directory and have been used in the notebook. The data for inference has been obtained as a subset of the data that has been linked [here](https://github.com/alexklwong/void-dataset/blob/master/README.md). Due to the **compressed** format of the data in openly available Google drive links, uncompressing the same for few inference examples is not recommended. Hence this OpenVINO™ tutorial *ships* with the required data.

The authors have published their work here:

> [Monocular Visual-Inertial Depth Estimation](https://arxiv.org/abs/2303.12134)  
> Diana Wofk, René Ranftl, Matthias Müller, Vladlen Koltun


## Notebook contents

The notebook contains a detailed tutorial of the visual-inertial depth estimation pipeline starting from downloading pre - trained models, pre - processing images and the models, converting the models to the [OpenVINO IR](https://docs.openvino.ai/latest/openvino_ir.html) representation for inference and finally running them against an image dedicated for inference.


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
