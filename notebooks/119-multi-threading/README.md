# Optimize the multiple-threading workload with OpenVINO™

This notebook demonstrates how to optimize the performance of multiple-threading logic with CPU scheduling.

There are commonly 2 kinds of multiple-threading workloads during model inference:

- **Multiple-requests**:
Running a model network many times and each inference request is independent, so these requests can be started in parallel. For example, we will run multiple requests of the recognition network to identify each detected object.
- **Multiple-networks**:
In another case, we have to run several different networks and concatenate the results of them for further postprocessing logic. Since the output of each model is independent of this workload, these networks' inference requests can be started synchronously or asynchronously. For example, an ADAS system can detect pedestrians and segment lanes at the same time.

The multi-threading logic for CPU inference is a matrix of multiple approaches related to the hardware platform, operating system, and application properties.

## Notebook Contents

The tutorial consists of 2 examples:

* Configuration and optimization on multiple-requests inference with CPU scheduling, including people detection and person feature recognition.
* Configuration and optimization on multiple-networks inference with CPU scheduling, including pedestrian detection and lane segmentation.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).