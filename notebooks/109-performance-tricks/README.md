# Introduction to Performance Tricks in OpenVINO™
This notebook shows the tips on how to optimize inference in OpenVINO™.   

There are different performance targets for different use case scenarios. For example, for video conferencing, it normally requires the latency to be as short as possible. While for high resolution video/image analysis, high throughput is normally the performance target. As a result, different optimization tricks should be applied to achieve different performance targets.
In this notebook, we’ll show a set of performance tricks for optimizing inferencing latency. 

## Notebook Contents

Performance tricks in OpenVINO for latency mode

This notebook demonstrates how to optimize the inferencing latency in OpenVINO™.  A set of optimization tricks, including model conversion with different data precision, “AUTO” device with latency mode, shared memory, asynchronous inferencing mode, inferencing with further configuration, inferencing on GPU, etc. are introduced.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
