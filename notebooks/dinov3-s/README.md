# DINOv3-s

This notebook demonstrates dense per-patch feature extraction with
[DINOv3](https://arxiv.org/abs/2508.10104) (*DINOv3*, Meta AI 2025) using the ViT-S/16 backbone
([facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m))
with OpenVINO.

The notebook does the following --

1. Loads the pretrained DINOv3 ViT-S/16 backbone (PyTorch, Hugging Face `transformers`) and
   wraps it so it emits **dense per-patch features** `(B, D, h, w)` — the representation DINOv3
   uses for dense tasks (segmentation, correspondence, PCA).
2. Converts it to **OpenVINO IR** at **FP32** and **FP16**.
3. Compresses it to **INT8** with **NNCF weight-only quantization (WOQ)**, a data-free recipe
   that needs no calibration set.
4. Runs inference on CPU / GPU and visualizes the dense features as **PCA-RGB**, **KMeans
   segmentation** and a per-patch **cosine-similarity** map against the PyTorch reference.
5. Evaluates each precision with **cosine similarity (mean and worst case per patch), MSE and
   MAE**, plus a latency / throughput benchmark.

For illustration, the notebook uses a single image from the **ImageNet / ImageNet-ReaL** family
(the benchmark DINOv3 reports on), downloaded once into `data/`. The backbone weights are the
LVD-1689M self-supervised checkpoints published by the DINOv3 authors on the Hugging Face Hub.

## Notebook Contents

- One-time conda environment setup (`dinov3s-env`), guarded so it never re-runs by default.
- Download the DINOv3 ViT-S/16 weights into `checkpoints/`, then build the dense-feature wrapper
  and the paper-faithful validation transform (resize → center-crop → normalize).
- Download the sample image into `data/`.
- Compute the PyTorch dense-feature reference.
- Convert to OpenVINO IR (FP32/FP16) and compress to INT8 with NNCF WOQ, into `ov_models/`.
- Select a device (Intel GPU if present, else CPU) and run each precision on the same input.
- Report mean / min cosine similarity, MSE and MAE against the PyTorch reference.
- Benchmark median latency and throughput per engine.
- Visualize PCA-RGB, segmentation and per-patch cosine similarity, torch vs each precision.
- Print a summary of fidelity, latency/throughput and IR sizes.

## Cached Artifacts

Everything the notebook fetches or produces is written next to it and reused, so a second run
goes straight to inference and prints what it skipped:

| folder | contents | re-run behaviour |
| --- | --- | --- |
| `checkpoints/` | DINOv3 ViT-S/16 PyTorch weights | download skipped if present |
| `data/` | the sample image | download skipped if present |
| `ov_models/` | `fp32/`, `fp16/`, `int8/` OpenVINO IRs | conversion skipped if present |

The IR filenames carry the input resolution, so changing `IMAGE_SIZE` triggers a fresh
conversion automatically.

## Installation Instructions

This is a self-contained example. It is recommended to run the notebook in a virtual environment, it only needs a Jupyter server to start. For general environment setup, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).

The DINOv3 repository on the Hugging Face Hub is **gated**: accept the licence on the
[model page](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) and authenticate
with `hf auth login` (or set `HF_TOKEN` in the configuration cell) before the first run.