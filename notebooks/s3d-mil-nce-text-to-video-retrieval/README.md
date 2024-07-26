# Text-to-Video retrieval with S3D MIL-NCE and OpenVINO

This tutorial based on [the TensorFlow tutorial](https://www.tensorflow.org/hub/tutorials/text_to_video_retrieval_with_s3d_milnce) that demonstrates how to use the [S3D MIL-NCE](https://tfhub.dev/deepmind/mil-nce/s3d/1) model from TensorFlow Hub to do text-to-video retrieval to find the most similar videos for a given text query.

MIL-NCE inherits from Multiple Instance Learning (MIL) and Noise Contrastive Estimation (NCE). The method is capable of addressing visually misaligned narrations from uncurated instructional videos. Two model variations are available with different 3D CNN backbones: I3D and S3D. In this tutorial we use S3D variation. More details about the training and the model can be found in [End-to-End Learning of Visual Representations from Uncurated Instructional Videos](https://arxiv.org/abs/1912.06430) paper.

This tutorial demonstrates step-by-step instructions on how to run and optimize S3D MIL-NCE model with OpenVINO. An additional part demonstrates how to run quantization with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up the inference.

## Notebook contents
This tutorial consists of the following steps:
- Prerequisites
- The original inference
- Convert the model to OpenVINO IR
- Compiling models
- Inference
- Model quantization

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/s3d-mil-nce-text-to-video-retrieval/README.md" />
