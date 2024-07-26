# Image Generation with Stable Diffusion using OpenVINO TorchDynamo backend

This notebook demonstrates how to use a **[Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1)** model for image generation with [OpenVINO TorchDynamo backend](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html). The `torch.compile` feature enables you to use OpenVINO for PyTorch-native applications. It speeds up PyTorch code by JIT-compiling it into optimized kernels.

By default, Torch code runs in eager-mode, but with the use of torch.compile it goes through the following steps:

1. Graph acquisition - the model is rewritten as blocks of subgraphs that are either:
    * compiled by TorchDynamo and “flattened”,
    * falling back to the eager-mode, due to unsupported Python constructs (like control-flow code).
2. Graph lowering - all PyTorch operations are decomposed into their constituent kernels specific to the chosen backend.
3. Graph compilation - the kernels call their corresponding low-level device-specific operations.

## Notebook Contents

This notebook demonstrates how to run stable diffusion using OpenVINO TorchDynamo backend.

Notebook contains the following steps:

1. Create PyTorch models pipeline using Diffusers library.
2. Import OpenVINO backend using `torch.compile`.
3. Run Stable Diffusion pipeline with OpenVINO TorchDynamo backend.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/stable-diffusion-torchdynamo-backend/README.md" />
