# Optimize the parallel workload with OpenVINO™

This notebook demonstrates how to optimize the parallelism performance of multiple-models and multiple-requests pipeline.

These are 2 kinds of parallelism workload during model inference:
    1. Same model network is run many times and each request is independent, so these requests can be started in parallel. For example: we can run multiple requests of recognition network to identify each detected object.
    2. Different model networks' outputs are independent, so we can run them in parallel as well. For example: an ADAS system can detect pedestrian and segment lane at same time.


## Notebook Contents

The tutorial consists of the 2 sub examples:

* Configuration and optimization on multiple-requests infernce scenario which includes the people detection and person's feature recongnition.
* Configuration and optimization on multiple-models infernce scenario which includes the pedestrian detection and lane segmentation.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).