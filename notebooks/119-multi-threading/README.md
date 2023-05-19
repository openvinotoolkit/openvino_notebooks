# Optimize the multiple-threading workload with OpenVINO™

This notebook demonstrates how to optimize the performance of multiple-threading logic with CPU scheduling.

These are commonly 2 kinds of multiple-threading workload during model inference:

- **Multiple-requests**:
Running a model network for many times and each inference request is independent, so these requests can be started in parallel. For example: we will run multiple requests of recognition network to identify each detected object.
- **Multiple-networks**:
In another case, we have to run several different networks and concatenate the results of them for further postprocess logic. Since the output of each model is independent in this workload, these networks' inference request can be started sychronously or asychronously. For example: an ADAS system can detect pedestrian and segment lane at same time.

The multi-threading logic for CPU inference is matrix of multiple approaches related to hardware platform, operating system, and application properties.

## Notebook Contents

The tutorial consists of 2 examples:

* Configuration and optimization on multiple-requests inference with CPU scheduling, including people detection and person's feature recongnition.
* Configuration and optimization on multiple-networks infernce with CPU scheduling, including pedestrian detection and lane segmentation.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).