# DINOv2 ViT-S/14 with OpenVINO (FP32 / FP16 / INT8)

This notebook demonstrates dense per-patch feature extraction with
[DINOv2](https://arxiv.org/abs/2304.07193) (*DINOv2*, Meta AI 2023) using the ViT-S/14 backbone
([facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small)) with OpenVINO.

The notebook does the following:

1. Loads the pretrained DINOv2 ViT-S/14 weights from the Hugging Face Hub into an **inline**
   architecture defined with plain `torch.nn` (no model library needed) and wraps it so it emits
   **dense per-patch features** `(B, D, h, w)` — the representation DINOv2 uses for dense tasks
   (segmentation, correspondence, PCA).
2. Converts it to **OpenVINO IR** at **FP32** and **FP16**.
3. Quantizes it to **INT8** with **NNCF post-training quantization (PTQ)** — a strided subset
   of ImageNet-val images calibrates the quantizers (MIXED preset, GPU target, transformer-aware).
   If the calibration set is not present, it falls back to **weights-only INT8**
   (`nncf.compress_weights`), so all three precisions still run end-to-end without any calibration
   data.
4. Runs inference on CPU / GPU and visualizes the dense features as **PCA-RGB**, **KMeans
   segmentation** and a per-patch **cosine-similarity** map against the PyTorch reference.
5. Evaluates each precision with **cosine similarity, MSE and MAE**, plus a median-latency
   benchmark per engine (PyTorch on CPU and, with an XPU build, on the Intel GPU).

For illustration, the notebook uses a single sample image, downloaded once into `data/`. The
backbone weights are the LVD-142M self-supervised checkpoints published by the DINOv2 authors on
the Hugging Face Hub.

## Notebook Contents

- Bootstrap a dedicated virtual environment (`.dinov2s-nb-venv`), register it as a Jupyter kernel
  and install the dependencies into the active kernel; every step is skipped when already done, so
  re-runs are instant.
- Download the DINOv2 ViT-S/14 weights into `checkpoints/`, then build the dense-feature wrapper
  and the paper-faithful validation transform (resize → center-crop → normalize).
- Download the sample image into `data/`.
- Compute the PyTorch dense-feature reference.
- Convert to OpenVINO IR (FP32/FP16) and quantize to INT8 with NNCF PTQ — or weights-only
  INT8 when no calibration set is available — into `ov_models/`.
- Select a device (Intel GPU if present, else CPU, with an automatic fallback if the GPU
  driver aborts device enumeration) and run each precision on the same input.
- Report cosine similarity, MSE and MAE against the PyTorch reference.
- Benchmark median latency per engine.
- Visualize PCA-RGB, segmentation and per-patch cosine similarity, torch vs each precision.
- Print a summary of fidelity, latency and IR sizes.

## Cached Artifacts

Everything the notebook fetches or produces is written next to it and reused, so a second run
goes straight to inference and prints what it skipped:

| folder | contents | re-run behaviour |
| --- | --- | --- |
| `checkpoints/` | DINOv2 ViT-S/14 PyTorch weights | download skipped if present |
| `data/` | the sample image | download skipped if present |
| `ov_models/` | `fp32/`, `fp16/`, `int8/` OpenVINO IRs | conversion skipped if present |

## INT8 Calibration Set

PTQ draws a strided subset of ImageNet-val images (all 1000 classes, no labels needed) from
`datasets/imagenet/val` — place the ImageNet-val class folders there, or point `CALIB_DIR` at
them in the configuration cell. If the set is not found, INT8 falls back to **weights-only
quantization** (no calibration data needed), so all three precisions still run end-to-end.

## Installation Instructions

This is a self-contained example. It is recommended to run the notebook in a virtual environment,
it only needs a Jupyter server to start; the first cell creates one (`.dinov2s-nb-venv`), installs
the dependencies — the XPU torch build on machines with an Intel GPU — and registers it as a
Jupyter kernel. For general environment setup, please refer to
[Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).

The [`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small) repository on the
Hugging Face Hub is **public**: no token or licence acceptance is needed before the first run.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/dinov2-s/README.md" />
