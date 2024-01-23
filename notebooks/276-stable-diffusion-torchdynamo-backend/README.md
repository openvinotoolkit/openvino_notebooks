# Image Generation with Stable Diffusion using OpenVINO TorchDynamo backend

This notebook demonstrates how to use a **[Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1)** model for image generation with [OpenVINO TorchDynamo backend](https://docs.openvino.ai/2023.1/pytorch_2_0_torch_compile.html). The `torch.compile` feature enables you to use OpenVINO for PyTorch-native applications. It speeds up PyTorch code by JIT-compiling it into optimized kernels.

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

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
