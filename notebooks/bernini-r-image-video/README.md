# Unified image & video generation with Bernini-R-1.3B and OpenVINO

[Bernini-R-1.3B](https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers) is a unified, multi-task diffusion *renderer* from ByteDance. A single model handles text-to-image, image editing, text-to-video, video editing, and reference-image-to-video generation. It is fine-tuned from [Wan2.1-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) and re-uses the Wan building blocks: a [`WanTransformer3DModel`](https://github.com/bytedance/Bernini) diffusion transformer (DiT), a `UMT5EncoderModel` text encoder, an `AutoencoderKLWan` spatio-temporal VAE, and a `UniPCMultistepScheduler` flow-matching scheduler.

You can find more details in the [model card](https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers) and the [original repository](https://github.com/bytedance/Bernini).

In this tutorial we convert, optimize and run Bernini-R with OpenVINO.

## How the pipeline is split for OpenVINO

Bernini's generation logic lives in a python denoising loop (`bernini.models.wan_diffusion.GEN_Wanx22.sample`) that selects among **seven guidance modes**, builds source-id rotary embeddings, assembles a variable number of conditioning tokens per step, and steps the scheduler. Static OpenVINO graphs cannot express that data-dependent control flow, so only the heavy *leaf* compute is pushed into OpenVINO while every loop and branch stays in python:

- **Text encoder** — a single static graph (token length fixed to 512).
- **Transformer** — the patch-embedding (a small `Conv3d`) and rotary-embedding construction stay in torch (so `patch_vae_latent` runs exactly as in the reference); the condition-embedder + 30 transformer blocks + output projection are exported as one graph (`BlocksCore`) with a **dynamic packed-token axis**, so the same compiled model serves every guidance combination (uncond / V / VI / VTI ...). The variable-length attention reduces to ordinary attention for single-sample inference and is implemented with `scaled_dot_product_attention`; rotary embeddings are applied with real arithmetic to keep complex ops out of the graph.
- **VAE encoder / decoder** — the Wan VAE walks the temporal axis with a python loop and a causal feature cache, so one graph is valid for a single temporal length. We export a graph per latent length (length `1` covers images / single frames; video lengths are also compiled lazily at run time).

The original `bernini` pipeline and sampler methods are then re-used **verbatim** by injecting these OpenVINO-backed leaf modules, which keeps the OpenVINO pipeline numerically aligned with the reference across all tasks and lets you pick a different inference device (CPU / GPU) per component.

`convert_pipeline` copies the configs, tokenizer and scheduler into the OpenVINO output directory, so the converted model is **self-contained**: `load_ov_pipeline` reads everything from that directory, compiles only OpenVINO IR, and does **not** load the original PyTorch model or any of its weights. (The single exception is the patch-embedding `Conv3d`, a few-KB tensor saved next to the IR, which must stay in torch to drive the data-dependent patchify + rotary embedding.) The original directory is only needed if you later request a video length whose VAE graph was not pre-built and want it converted on the fly.

> **Device placement defaults**: the transformer (run several times per step) defaults to `AUTO` so it can use the GPU; the **text encoder defaults to CPU** and the **VAE to `AUTO`** with an automatic per-call fallback.
> - The UMT5 text encoder must run in fp32 — it overflows in fp16, producing a black image — and compiling the ~5 GB fp32 model on GPU takes ~80 s and competes for GPU memory, whereas on CPU it compiles in ~2 s and only runs twice per generation.
> - The VAE runs an **image** (single-frame) decode on the GPU (≈1 GB, ~6.7× faster than CPU), but a **video** decode unrolls the Wan causal-conv loop into a tens-of-GB activation buffer that exhausts GPU memory (confirmed on both stable and nightly OpenVINO, integrated GPU). So `OVVAEWrapper` routes any VAE decode/encode whose latent temporal length exceeds `vae_gpu_max_latent_frames` (default 1) to CPU automatically — you get the GPU speed-up for images with no out-of-memory risk for video.
> If you place the text encoder / VAE on GPU, `load_ov_pipeline` automatically applies an fp32 precision hint to keep them numerically stable; pass `ov_config` to override.

The helpers are integrated into a single file, [ov_bernini_helper.py](ov_bernini_helper.py): model conversion, the OpenVINO wrapper modules, and `load_ov_pipeline`.

## Notebook contents
This tutorial consists of the following steps:
- Prerequisites
- Convert and Optimize the model (FP16 / INT8 / INT4 weight compression)
- Run the inference pipeline (text-to-image and text-to-video)
- Interactive inference with Gradio

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

> **Note**: the reference Bernini implementation pins `diffusers==0.35.2` and `transformers==4.57.3`; the first notebook cell installs these and the `bernini` package. Restart the kernel after the install cell if these versions differ from your base environment.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/bernini-r-image-video/README.md" />
