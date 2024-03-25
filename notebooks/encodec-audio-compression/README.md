# Audio compression with EnCodec and OpenVINO

Compression is an important part of the Internet today because it enables people to easily share high-quality photos, listen to audio messages, stream their favorite shows, and so much more. Even when using today’s state-of-the-art techniques, enjoying these rich multimedia experiences requires a high speed Internet connection and plenty of storage space. AI helps to overcome these limitations: "Imagine listening to a friend’s audio message in an area with low connectivity and not having it stall or glitch."

In this tutorial, we consider how to use OpenVINO and EnCodec algorithm for hyper compression of audio.
EnCodec is a real-time, high-fidelity audio codec that uses AI to compress audio files without losing quality. It was introduced in [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) paper by Meta AI. More details about this approach can be found in [Meta AI blog](https://ai.facebook.com/blog/ai-powered-audio-compression-technique/) and original [repo](https://github.com/facebookresearch/encodec).


## Notebook Contents

This notebook demonstrates how to convert and run EnCodec model using OpenVINO.

Notebook contains the following steps:
1. Instantiate and run an EnCodec audio compression pipeline.
2. Convert models to OpenVINO IR format, using model conversion API.
3. Integrate OpenVINO to the EnCodec pipeline.

As the result, we get a pipeline that accepts input audio file and converts it to compressed representation, ready for being saved on disk or sent to a recipient. After that, it can be successfully decompressed back to audio.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).