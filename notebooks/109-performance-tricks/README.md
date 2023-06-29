# Introduction to Performance Tricks in OpenVINO™

These notebooks show tips on how to optimize inference in OpenVINO™.   

There are different performance targets for various use case scenarios. For example, video conferencing usually requires the latency to be as short as possible. While for high-resolution video/image analysis, high throughput is typically the performance target. As a result, different optimization tricks should be applied to achieve other performance targets.
In this notebook, we’ll show a set of performance tricks for optimizing inferencing latency. 

## Notebook Contents

1. Performance tricks in OpenVINO for latency mode

This notebook demonstrates how to optimize the inferencing latency in OpenVINO™.  A set of optimization tricks, including model conversion with different data precision, “AUTO” device with latency mode, shared memory, asynchronous inferencing mode, inferencing with a further configuration, inferencing on GPU, etc., are introduced.

![](https://user-images.githubusercontent.com/4547501/229120774-01f4f972-424d-4280-8395-220dd432985a.png)

2. Performance tricks in OpenVINO for throughput mode

TBA

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
