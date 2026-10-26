# DINOv2 ViT-S/14-Reg with OpenVINO (FP32 / FP16 / INT8)

This notebook demonstrates dense per-patch feature extraction with
[DINOv2](https://arxiv.org/abs/2304.07193) (*DINOv2*, Meta AI 2023) using the ViT-S/14 backbone
with 4 register tokens
([facebook/dinov2-with-registers-small](https://huggingface.co/facebook/dinov2-with-registers-small))
with OpenVINO.

The notebook does the following:

1. Loads the pretrained DINOv2 ViT-S/14-Reg backbone (PyTorch, Hugging Face `transformers`) and
   wraps it so it emits **dense per-patch features** `(B, D, h, w)` — the representation DINOv2
   uses for dense tasks (segmentation, correspondence, PCA).
2. Converts it to **OpenVINO IR** at **FP32** and **FP16**.
3. Compresses it to **INT8** with **NNCF post-training quantization (PTQ)** — a strided subset of
   a local calibration set calibrates the quantizers (MIXED preset, GPU target,
   transformer-aware) — and always produces **weights-only INT8** (`nncf.compress_weights`)
   alongside it, so an INT8 model is available even without any calibration data.
4. Runs inference on CPU / GPU and visualizes the dense features as **PCA-RGB**, **k-means
   segmentation** and a per-patch **cosine-similarity** map against the PyTorch reference.
5. Evaluates each precision with **cosine similarity, mean squared error and MAE**, plus a
   median-latency benchmark per engine.

For illustration, the notebook uses a single demo image — a local COCO `val2017` image when
that folder is present, otherwise the well-known COCO dog image used across the DINO examples —
downloaded once into `data/`. The backbone weights are the self-supervised checkpoints
published by the DINOv2 authors on the Hugging Face Hub.

## Notebook Contents

- Install dependencies into the active Jupyter kernel (a dedicated virtual environment,
  `.dinov2sreg-nb-venv`); re-runs are fast because pip skips packages that are already
  satisfied.
- Download the DINOv2 ViT-S/14-Reg weights into `checkpoints/`, then build the dense-feature
  wrapper and the paper-faithful validation transform (resize → center-crop → normalize).
- Download the sample image into `data/`.
- Compute the PyTorch dense-feature reference.
- Convert to OpenVINO IR (FP32/FP16) and quantize to INT8 with NNCF PTQ — plus weights-only
  INT8, which needs no calibration data — into `ov_models/`.
- Select a device (Intel GPU if present, else CPU, with an automatic fallback if the GPU
  driver aborts device enumeration) and run each precision on the same input.
- Report cosine similarity, mean squared error and MAE against the PyTorch reference.
- Benchmark median latency per engine (torch on CPU, and on the Intel GPU through `torch.xpu`
  when the torch build provides it).
- Visualize PCA-RGB, segmentation and per-patch cosine similarity, torch vs each precision.
- Print a summary of fidelity, latency and IR sizes.

## Cached Artifacts

Everything the notebook fetches or produces is written next to it and reused, so a second run
goes straight to inference and prints what it skipped:

| folder | contents | re-run behaviour |
| --- | --- | --- |
| `checkpoints/` | DINOv2 ViT-S/14-Reg PyTorch weights | download skipped if present |
| `data/` | the sample image | download skipped if present |
| `ov_models/` | `fp32/`, `fp16/`, `int8-ptq/`, `int8-woq/` OpenVINO IRs | conversion skipped if present |

The IR filenames carry the input resolution, so changing `IMAGE_SIZE` triggers a fresh
conversion automatically.

## INT8 Calibration Set

PTQ draws a strided 512-image subset (no labels needed) from the first folder that exists:
`datasets/imagenet/val` (ImageNet-val class folders, spanning all 1000 classes) or
`datasets/coco/val2017` — place your images there, or point `CALIB_DIR` at them in the
configuration cell. Nothing is downloaded for calibration. If no set is found, PTQ is skipped
and the notebook still produces FP32, FP16 and weights-only INT8, so the pipeline always runs
end-to-end.

## Installation Instructions

This is a self-contained example. It is recommended to run the notebook in a virtual
environment, it only needs a Jupyter server to start. For general environment setup, please
refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).

The `dinov2-with-registers-small` repository on the Hugging Face Hub is **not gated** — no
licence acceptance or token is required. On a machine with an Intel GPU, the first cell
installs the Intel GPU (XPU) PyTorch build so the latency benchmark also reports a `torch.xpu`
row; elsewhere the CPU build is used.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/dinov2-s-reg/README.md" />
