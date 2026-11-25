# Synchronized audio-video generation with LTX-2.3 and OpenVINO(TM)

[LTX-2.3](https://huggingface.co/diffusers/LTX-2.3-Diffusers) is a diffusion transformer model that generates video and synchronized audio in a single pipeline. The prompt can describe both the visual scene and its soundtrack, including ambient sounds, music, and speech.

https://github.com/user-attachments/assets/0a297ba9-579d-4867-8e9d-b65eaead7ad1

*Example from the original notebook: "A cat stretches lazily on a sunny windowsill, purring softly, while birds chirp outside." with the INT8 model on CPU (768x512, 121 frames, with audio).*

This tutorial shows how to export LTX-2.3 to OpenVINO Intermediate Representation with [Optimum Intel](https://huggingface.co/docs/optimum/intel/index), compress its weights with [NNCF](https://github.com/openvinotoolkit/nncf), and run one of two workflows:

- **Text-to-audio-video** generates video and audio from a text prompt with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) `Text2VideoPipeline` or with the Optimum Intel pipeline.
- **Image-to-audio-video** uses an initial image together with a text prompt and runs with the Optimum Intel pipeline. The default road image is paired with a traffic prompt; adapt the prompt to your image and desired sounds.

A single **text-to-video** export is reused by both workflows. It identifies `LTX2Pipeline` for GenAI and also includes the VAE encoder for Optimum Intel image conditioning. The current exporter moves RoPE calculations outside the transformer and connectors; a new output directory avoids reusing earlier IRs.

LTX-2.3 is a large model. Download, export, compression, and inference require substantial disk space and memory.

Model weights are subject to the [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2). Review its terms before use.

## Component support

- **Optimum Intel:** the notebook pins a `main` revision with [external RoPE inputs](https://github.com/huggingface/optimum-intel/pull/2070), the [group-wise INT8 recipe](https://github.com/huggingface/optimum-intel/pull/2076), and the [updated INT4 recipe](https://github.com/huggingface/optimum-intel/pull/2127). These updates are not in PyPI 2.2.0.
- **OpenVINO GenAI:** [LTX-2.3 text-to-audio-video](https://github.com/openvinotoolkit/openvino.genai/pull/4559) and [external RoPE support](https://github.com/openvinotoolkit/openvino.genai/pull/4692) are required; PyPI 2026.4.1 does not include them. The requirements stay within the 2026.5 series (`>=2026.5.0.dev20261116,<2026.6`), accepting a future stable 2026.5.0 release but excluding 2027.0 nightlies. Image-to-audio-video still uses Optimum Intel.
- **OpenVINO and OpenVINO Tokenizers:** installed together with GenAI from the nightly index for compatible runtime, frontend, device plugins, and tokenizer extension. CPU and GPU are selectable; NPU is not exposed.
- **NNCF:** the notebook pins the [merged group-wise INT8 implementation](https://github.com/openvinotoolkit/nncf/pull/4278), which is not in PyPI 3.4.0. The source build uses explicitly installed `setuptools>=77.0` and `wheel` without build isolation. Direct NNCF compression avoids the CLI fallback to per-channel INT8 when the installed version is below the future 3.5.0 release, including development builds.
- **Diffusers / Transformers:** kept at 0.40.0 / 5.10.4 for the API used by this notebook.

INT8 uses symmetric group-size-32 weights for the transformer and connectors, excluding sensitive layers, and per-channel asymmetric weights for the text encoder. INT4 now uses the model-specific CLI recipe: symmetric group-size-32 transformer and connector weights, with an asymmetric group-size-128 text encoder. INT4 remains experimental here; previous similarity measurements do not validate this changed recipe.

The notebook covers the full `diffusers/LTX-2.3-Diffusers` checkpoint only. Distilled models, LTX-2.5, two-stage upsampling, arbitrary conditions, and LoRA adapters are outside its scope. Selectable devices and upstream tiny-model tests do not establish full-checkpoint quality or memory requirements on every device.

## Notebook contents

- Prerequisites
- Convert the model to OpenVINO IR
- Compress model weights
- Run audio-video generation
- Split GenAI stages across CPU/GPU, optionally with a denoising precision hint
- Optional reshape-before-compile and compiled-model caching with either backend
- Video/audio CFG controls and fast CFG-only generation
- GenAI cancellation and stage timings in the interactive demo
- Interactive inference
- Limitations

CPU is selected by default and AUTO is excluded. In the GenAI split-device mode, the text encoder and connectors, transformer, and video/audio decoders have separate device selectors; CPU / GPU / CPU keeps text encoding and decoding off the GPU.

Static shapes fix the clip dimensions, frame rate, and CFG batch size, including in the demo; reload the pipeline to change them. Optimum Intel retains some dynamic token dimensions and requires CFG on its static path. `CACHE_DIR` is enabled by default for both backends. Fast mode disables STG and modality-isolation passes but retains CFG; it can change output quality. GenAI cancellation is checked between denoising steps and leaves the pipeline available for the next request.

Existing model directories are reused. Regenerate INT8 output created with an earlier notebook revision from the preserved FP16 export to apply the serialization that retains sensitive FP32 constants.

## Validation scope

Notebook execution is currently skipped in public CI on all three operating systems; passing metadata and code checks do not prove inference. The patched tiny-model copy uses per-channel INT8 without the full-model ignored scope because its layer dimensions do not support the group-size-32 recipe. That smoke path does not validate the full checkpoint's compression quality.

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/ltx2.3-audio-video/README.md" />
