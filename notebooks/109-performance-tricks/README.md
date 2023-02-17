# Introduction to Performance Tricks in OpenVINO™
Option1: This notebook shows the tips on how to optimize inference in OpenVINO™.  

Option2: This folder includes 2 notebooks with tips on how to optimize inference in OpenVINO™. The first one shows the performances tricks for latency mode, while the 2nd notebook for throughput mode will be posted soon. 

There are different performance targets for different use case scenarios. For example, for video conferencing, it normally requires the latency to be as short as possible. While for high resolution video/image analysis, high throughput is normally the performance target. 
Therefore, in this (or the first?) notebook, we’ll show a set of performance tricks for optimizing inferencing latency. 

## Notebook Contents

Performance tricks in OpenVINO for latency mode

This notebook demonstrates how to optimize the inferencing latency in OpenVINO™.  A set of optimization tricks, including model conversion with different data precision, prepostprocessing API, “AUTO” device with latency mode, shared memory, asynchronous inferencing mode, etc. are introduced.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
