# Wav2Lip: Accurately Lip-syncing Videos and OpenVINO

Lip sync technologies are widely used for digital human use cases, which enhance the user experience in dialog scenarios.

[Wav2Lip](https://github.com/Rudrabha/Wav2Lip) is a novel approach to generate accurate 2D lip-synced videos in the wild with only one video and an audio clip. Wav2Lip leverages an accurate lip-sync “expert" model and consecutive face frames for accurate, natural lip motion generation.

![teaser](https://github.com/user-attachments/assets/11d2fb00-4b5a-45f3-b13b-49636b0d48b1)

In this notebook, we introduce how to enable and optimize Wav2Lippipeline with OpenVINO. This is adaptation of the blog article [Enable 2D Lip Sync Wav2Lip Pipeline with OpenVINO Runtime](https://blog.openvino.ai/blog-posts/enable-2d-lip-sync-wav2lip-pipeline-with-openvino-runtime).

Here is Wav2Lip pipeline overview:

![wav2lip_pipeline](https://cdn.prod.website-files.com/62c72c77b482b372ac273024/669487bc70c2767fbb9b6c8e_wav2lip_pipeline.png)

## Notebook contents
The tutorial consists from following steps:

- Prerequisites
- Convert the original model to OpenVINO Intermediate Representation (IR) format
- Compiling models and prepare pipeline
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/wav2lip/README.md" />
