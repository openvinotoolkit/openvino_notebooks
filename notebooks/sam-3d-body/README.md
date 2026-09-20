# SAM 3D Body

This notebook demonstrates 3D human pose and shape estimation with
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (*SAM 3D Body*, Meta AI 2025)
using the DINOv3 ViT-H/16+ backbone
([facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3))
with OpenVINO.

Given an image and a person bounding box, the model predicts 70 2D/3D keypoints and a full
parametric body mesh (MHR). The notebook does the following --

1. Loads the reference PyTorch checkpoint and runs it on **CPU** or **Intel XPU** to produce a
   full-precision reference (keypoints, mesh, PCK@0.05, latency).
2. Converts the pipeline to **OpenVINO IR** at **FP16** and **INT8**.
3. Compresses the INT8 variant with **NNCF weight-only quantization (WOQ)**, a data-free recipe
   that needs no calibration set.
4. Runs both IRs on the **Intel GPU** and reproduces the iterative MHR feedback loop in pure
   NumPy + OpenVINO (no PyTorch at inference time).
5. Verifies that the converted models agree with the PyTorch reference — visually (skeletons and
   mesh overlays) and numerically (PCK@0.05 and per-keypoint pixel error).

For illustration, the notebook uses a single image from **COCO val2017** — one person with all
17 keypoints visible and a large bounding box, which makes for a clean comparison. The
ground-truth box is fed to the model, so the pose accuracy is measured without a detector in the
loop. Nothing is bundled: the image is downloaded from the public COCO servers on first use and
its ground-truth annotation is embedded in the notebook. The backbone weights are the
DINOv3-H+ checkpoints published by the SAM 3D Body authors on the Hugging Face Hub.

## Notebook Contents

- Kernel check plus a dependency-install cell — no environment file, no manual setup.
- Device discovery for OpenVINO and PyTorch, plus a single configuration cell for all knobs.
- Fetch the `sam_3d_body` package from the official GitHub repo into the root folder (skipped if present).
- Download the gated SAM 3D Body checkpoint into `checkpoints/` (skipped if present).
- Download the COCO sample image and draw the ground-truth box and keypoints.
- Run the PyTorch reference on CPU or XPU and record PCK@0.05 and latency.
- Convert to OpenVINO IR (FP16/INT8) into `ov_models/`, then report the on-disk footprint.
- Compile each IR on the Intel GPU, warm up, and run inference on the same person.
- Compare all backends side by side — skeletons, mesh overlays, and a PCK/latency table.

## Cached Artifacts

Everything the notebook fetches or produces is written next to it and reused, so a second run
goes straight to inference and prints what it skipped:

| folder           | contents                                                       | re-run behaviour              |
| ---------------- | -------------------------------------------------------------- | ----------------------------- |
| `sam_3d_body/` | the `sam_3d_body` package, fetched from the official GitHub repo | download skipped if present   |
| `checkpoints/` | SAM 3D Body PyTorch weights (`model.ckpt`, `mhr_model.pt`) | download skipped if present   |
| `sample_data/` | the demo image, downloaded from the COCO servers                | download skipped if present   |
| `ov_models/`   | `fp16/`, `int8/` OpenVINO IRs                              | conversion skipped if present |

Set `FORCE_EXPORT = True` in the configuration cell (or delete `ov_models/<precision>/`) to
force a fresh conversion.

## Installation Instructions

This is a self-contained example that relies solely on its own code. There is **no
`requirements.txt`** and no conda environment — everything runs from a plain **`venv`
created inside this folder** (`.venv/`).

```bash
cd notebooks   # this folder
python -m venv .venv

.venv/bin/pip install ipykernel
.venv/bin/python -m ipykernel install --user --name sam3dbody-venv \
    --display-name "Python (sam3d-body-nb .venv)"
```

Then open [`sam3dbody.ipynb`](sam3dbody.ipynb) and select the **Python (sam3d-body-nb
.venv)** kernel from the kernel picker — the notebook's saved metadata already points at
this kernel name, so most Jupyter front-ends preselect it automatically once it exists.
The Prerequisites cells install every dependency into that same `.venv`; once installed,
re-running the notebook does not reinstall anything.

> Dependencies are installed with `sys.executable -m pip install`, so they always land in
> the environment backing the selected kernel — never in the environment that launched the
> Jupyter server, and never in the system Python. The first cell prints the interpreter it
> will install into and warns if that interpreter is not an isolated environment.

The SAM 3D Body repository on the Hugging Face Hub is **gated**: accept the licence on the
[model page](https://huggingface.co/facebook/sam-3d-body-dinov3) and authenticate with
`hf auth login` (or set `HF_TOKEN`) before the first run.

## Repository layout

```
notebooks/
├── sam3dbody.ipynb        # the notebook (entry point)
├── sam3d_data.py          # sample loading, PCK scoring, skeleton + mesh rendering
├── sam3d_ov.py            # OpenVINO IR runtime
├── sam3d_torch.py         # PyTorch reference inference + PyTorch → OpenVINO export
├── LICENSE                # the SAM License (governs the fetched package + weights)
├── NOTICE                 # how this project relates to SAM 3D Body
├── .venv/                 # local virtual environment (created by you, see above)
├── sam_3d_body/           # the sam_3d_body package (downloaded on first run)
├── checkpoints/           # reference checkpoint (downloaded on first run)
│   └── sam-3d-body-dinov3/
│       ├── model.ckpt     #   2.0 GB
│       └── assets/mhr_model.pt   # 664 MB
├── ov_models/             # exported OpenVINO IR (created on first run)
│   ├── fp16/
│   └── int8/
└── sample_data/           # demo image cache (COCO val2017, downloaded on first run)
```

## Configuration

All knobs live in the *Configuration* cell:

| Variable          | Default              | Meaning                                                                    |
| ----------------- | -------------------- | -------------------------------------------------------------------------- |
| `TORCH_DEVICE`  | `"cpu"`            | PyTorch reference backend:`"cpu"` or `"xpu"` (needs a torch+xpu build) |
| `OV_DEVICE`     | `"GPU"`            | OpenVINO target device; falls back to`"CPU"` if not found                |
| `PRECISIONS`    | `["fp16", "int8"]` | IR precisions to export and evaluate                                       |
| `PCK_THRESHOLD` | `0.05`             | PCK tolerance as a fraction of the GT bbox diagonal       |
| `FORCE_EXPORT`  | `False`            | `True` re-exports the IR even if it already exists                |
| `SAMPLE_NAME`   | `"000000368212"`   | Bundled sample to evaluate                                                 |

## Exporting the OpenVINO IR

```bash
python sam3d_torch.py \
    --precision fp16 int8 \
    --output_dir ov_models \
    --checkpoint checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

Each precision writes to `ov_models/<precision>/`. The FP16 export uses OpenVINO's built-in
`compress_to_fp16` flag and the INT8 export needs `nncf` (installed by the Prerequisites cell).

## Using the OpenVINO runtime on a sample image

```python
import cv2
import sam3d_data as data
from sam3d_ov import Sam3DBodyOpenVINO

pipe = Sam3DBodyOpenVINO("ov_models/fp16", device="GPU", precision="fp16")
pipe.warmup()  # JIT-compile GPU kernels so the first inference is steady-state

sample = data.make_sample(
    cv2.imread("my_image.jpg"),
    {"bbox": [x, y, w, h], "keypoints": [0] * 51},  # COCO-style annotation
)

out = pipe.infer_single(
    sample["img_rgb"], sample["bbox_xyxy"], focal_length=sample["focal_length"]
)
# out: j2d [70, 2], j3d [70, 3], verts [V, 3], cam_t [3], focal_length

person = {"keypoints_2d": out["j2d"], "bbox": sample["bbox_xyxy"]}
data.show_row([("prediction", data.render_views(sample["img_bgr"], person, None)[0])])
```

Notes:

- `bbox` is `[x1, y1, x2, y2]` in image pixels; `data.make_sample` converts a COCO
  `[x, y, w, h]` annotation for you.
- `focal_length` defaults to the image diagonal — the same assumption the model makes when no
  FOV estimator is in the loop.
- A persistent kernel cache lives in `<model_dir>/cache`; later runs load pre-compiled GPU
  kernels from disk, removing cold-start latency.

## Precision and quantization details

- **FP16** — weights are stored as FP16 (`ov.save_model(..., compress_to_fp16=True)`).
- **INT8** — NNCF *weight-only* compression via
  `nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM)`: asymmetric 8-bit
  weights, per-channel scales (`group_size=-1`), no calibration data, activations stay floating
  point.

## Evaluation

- **PCK@0.05** — the fraction of visible GT keypoints whose predicted position is within 5% of
  the GT bounding-box diagonal.
- **Latency** — time taken for a single `infer` call after warm-up.
- **Visual agreement** — 2D skeletons and 3D mesh overlays rendered side by side for every
  backend.

## Troubleshooting

- **`WARNING: 'GPU' not found ... falling back to CPU`** — install/update the OpenVINO GPU
  driver and runtime for your GPU generation, then restart the kernel.
- **Checkpoint download fails with a 401/403** — the repo is gated; request access on the
  Hugging Face model page and run `hf auth login`.
- **`XPU requested but unavailable`** — you need a `torch+xpu` build
  (`pip install torch --extra-index-url https://download.pytorch.org/whl/xpu`) and an Intel Arc
  GPU; otherwise leave `TORCH_DEVICE = "cpu"`.
- **Mesh rendering fails on a headless server** — ensure an EGL-capable OpenGL stack is
  installed (e.g. `libegl1-mesa`); the notebook already sets `PYOPENGL_PLATFORM=egl` before
  `pyrender` is imported.
- **NumPy import errors in OpenVINO/NNCF** — keep `numpy<2` (pinned by the Prerequisites cell);
  OpenVINO and NNCF are not yet NumPy-2 clean.
- **Slow first OpenVINO inference** — expected: the first call JIT-compiles GPU kernels. The
  notebook warms up before timing, and the kernel cache (`ov_models/<precision>/cache`) makes
  subsequent runs fast.
- **Re-export after changing the model** — set `FORCE_EXPORT = True` in the Configuration cell,
  or delete `ov_models/<precision>/`.

## Acknowledgements

- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)
- [OpenVINO](https://docs.openvino.ai/)
- [NNCF](https://github.com/openvinotoolkit/nncf)
- [COCO](http://cocodataset.org/)
