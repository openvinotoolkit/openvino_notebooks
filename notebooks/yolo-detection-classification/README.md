# YOLO Detection & Classification with Device Selection

This notebook demonstrates how to build an object detection and classification pipline using a YOLO model with OpenVINO™. It also highlights how to select and run inference on different hardware devices.

![Sample detection result](grocery_detect.jpg)

## Notebook Contents

The notebook is organized into the following sections:

1. **Environment Setup**  
   Verify installation of OpenVINO and required packages.

2. **Basic Inference**  
   - Download a YOLO model for object detection.  
   - Run detection without OpenVINO  


3. **OpenVINO Inferenece**  
   - Convert to OpenVINO format
   - Run on selected devices - CPU, GPU, NPU

4. **Crop Images**  
   - Extract detected objects as single images

5. **Classification of Single Images**  
   - Run classification on cropped images  

6. **Create full pipeline**  
   - Combine Detection, Cropping, Classification into single pipeline

7. **Compare Performance**
   - Compare performance for pipeline for CPU and GPU inference  


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

> **Note:** After installing `ipywidgets` and `jupyterlab_widgets`, you may need to **restart the Jupyter server** (not just the kernel) to enable widget rendering in JupyterLab.