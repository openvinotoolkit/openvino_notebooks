# LTX-Video 2.3 and OpenVINO™

This notebook demonstrates how to load [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) in Diffusers format, convert the main pipeline components to OpenVINO™ IR, and run CPU/GPU smoke checks for the exported models.

## Notebook contents

This tutorial covers the following steps:
- Load `dg845/LTX-2.3-Diffusers` and inspect model configuration.
- Convert the transformer, text encoder, and VAE encoder/decoder to OpenVINO IR.
- Validate compilation and smoke-test inference on CPU and GPU where available.
- Save reference outputs and summarize the current end-to-end pipeline status.

## Installation instructions

This is a self-contained example that installs its Python dependencies in the notebook.
For environment setup details, refer to the main [Installation Guide](../../README.md).

## Notes

- The notebook is focused on component conversion and validation for LTX-Video 2.3.
- Full end-to-end video generation pipeline wiring remains experimental and requires additional validation.
- Model weights are subject to the [LTX-2 Community License Agreement](https://huggingface.co/Lightricks/LTX-2.3).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/ltx-video-2.3/README.md" />
