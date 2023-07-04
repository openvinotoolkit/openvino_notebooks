# RAFT Optical Flow Estimation with OpenVINO™

This tutorial explains converting the RAFT ONNX\* model with OpenVNO and provides the image and video inference demo.

[Recurrent All-Pairs Field Transforms (RAFT) ](https://github.com/princeton-vl/RAFT) is a new deep network architecture for optical flow. 

Optical flow has applications in many video editing tasks, such as stabilization, compression, slow motion, etc. In addition,  some tracking and action recognition systems also use optical flow data.

FlowNet and RAFT are popular deep learning-based methods for motion estimation using optical flow. Compared to FlowNet, the first CNN method for calculating optical flow, RAFT is currently one of the most advanced methods.

RAFT extracts per-pixel features, builds multi-scale 4D correlation volumes for all pairs of pixels, and iteratively updates a flow field through a recurrent unit that performs lookups on the correlation volumes. RAFT achieves state-of-the-art performance.

![image](https://github.com/openvinotoolkit/openvino_notebooks/assets/102195992/c52394f9-d654-4224-98d1-0a9244527be9)

This tutorial demonstrates step-by-step instructions on how to run and optimize RAFT with OpenVINO based on the [ONNX-RAFT-Optical-Flow-Estimation](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/tree/main). 

[PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/04f86a550e2ce1c5fb70cfafdfdeb568700f61c2/252_RAFT) converted the original PyTorch models to ONNX models with different input sizes and iteration times. The compressed model's size is 227M, containing pre-trained models for diverse datasets (SINTEL, KITTI, CHAIRS, etc.).

## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run and optimize ONNX\* RAFT with OpenVINO.

The tutorial consists of the following steps:
- Download the ONNX model
- Convert ONNX model to OpenVINO IR
- Evaluate the performance
- Test on images and video

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

## Image Inference Examples

![image](https://github.com/openvinotoolkit/openvino_notebooks/assets/102195992/f73b5380-c37b-42df-b8fd-647dc904b58b)

![image](https://github.com/openvinotoolkit/openvino_notebooks/assets/102195992/7d9446b4-9bd0-4b20-a5aa-986bfaeb8f35)

## Video Inference Examples
Coco Walking in Berkeley
![Coco Walking in Berkeley](https://github.com/openvinotoolkit/openvino_notebooks/assets/102195992/830d2fca-623d-4182-a170-7cebbb739b1c)

Water Drop
![water-drop](https://github.com/openvinotoolkit/openvino_notebooks/assets/102195992/32292712-5509-461a-b46a-31647638a266)