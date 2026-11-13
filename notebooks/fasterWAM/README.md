# FasterWAM on OpenVINO: convert and run a world-action model

[FasterWAM](https://huggingface.co/hustvl/FasterWAM) is a world-action model for robot manipulation: one
camera observation, one language instruction and the proprioceptive state of the arm go in, the next
chunk of 32 low-level actions comes out.

This notebook exports the three graphs that make up one action chunk — a VAE encoder, a 30-block
video expert that returns fused K/V tensors, and a small action expert that is applied ten times
per chunk — and runs the complete pipeline on a CPU or an Intel GPU. Language is an input too, so the
**umT5-xxl text encoder** that turns the instruction into the 128-token context those graphs start
from is exported and run here as well: a fourth graph, with its own precision rule (fp16 weights, bf16
execution on a GPU, f32 on the CPU) and its own cache directory. It clones the FasterWAM source and
downloads the released checkpoint itself, into its `model/` folder, exports the IR into `model/ir/`
(fp16 by default, with an optional weight-only INT8 export) and keeps it for later runs.

## Notebook Contents

- the three-graph split, and why the video expert runs once while the action expert runs ten times
- a self-fetched world: a shallow `git clone` of [hustvl/FasterWAM](https://github.com/hustvl/FasterWAM)
  and the released LIBERO checkpoint from Hugging Face, both pinned to the revision the notebook was
  validated with and stored in the notebook's `model/` folder; the exported IR lands in `model/ir/`
- converting to OpenVINO IR with the exporter bundled in `utils/fwam_export`
  (Torch → ONNX → OpenVINO IR) as a subprocess — a converted set is kept in `model/ir/` and reused, never deleted; the notebook kernel itself never imports Torch
- a per-stage latency breakdown (VAE, video prefill, K/V cast, action loop) of the compiled pipeline
- an accuracy check of the converted graphs against the original PyTorch model run on the CPU, per graph
- the text encoder: the instruction is tokenized and run through umT5-xxl (bf16 on a GPU, f32 on the
  CPU), the resulting 128-token context is checked against the contexts in `artifacts/`, and the encoder
  follows the device you pick: bf16 on a GPU, the exact f32 arm on the CPU
- an interactive playground over the same 32 recorded samples: pick any observation (and therefore
  any instruction) or type an instruction of your own, which is encoded by the text encoder above,
  replace the recorded diffusion noise with your own seed, and change the Euler
  step count and scheduler shift live — the predicted chunk is overlaid on the expert chunk, next to
  that chunk's own per-stage timings, without recompiling anything

## Requirements

| Peak memory | fp16 | INT8 (weight-only) |
|---|---|---|
| **CPU** - export (once) | about 56 GB | about 56 GB |
| **CPU** - inference (graphs + text encoder, f32) | about 54 GB | about 43 GB |
| **GPU** - export (once, runs on the CPU) | about 56 GB | about 56 GB |
| **GPU** - inference (graphs + text encoder, bf16) | about 19 GB | about 10 GB |

Export always runs on the CPU, whatever device you pick, and peaks while the video unit is
converted; the text encoder's conversion and the one-off PyTorch reference chunk stay below that.
Inference counts the three graphs and the text encoder loaded together. Below about 64 GB of RAM, run
the CPU with the INT8 graphs.

The first run also needs network and disk: it downloads about 28 GB - the repository clone, the 11.1 GB
checkpoint, the 11.4 GB text encoder and 4.7 GB of LIBERO episodes - and holds about 75 GB at the peak
of the conversion, mostly the ONNX scratch and the IR itself. Later runs reuse all of it.


## Installation Instructions

The evaluation artifacts (`artifacts/`) are generated on first use, by the cell right before the sample
inputs: four of the five steps derive everything from the released checkpoint and the umT5 weights, and
the observations come from the LIBERO episodes, which that cell downloads from the Hugging Face dataset
[`yuanty/LIBERO-fastwam`](https://huggingface.co/datasets/yuanty/LIBERO-fastwam) (CC BY 4.0) - 4.7 GB, the
already-unpacked `lerobot_v30` tree (the repository's tarballs hold the same bytes again and are
skipped). Nothing else needs to be set up, and both the episodes and the packs are reused afterwards.

If your OpenVINO build is a source build rather than a `pip install`, make it importable for the
kernel and start Jupyter from that
environment. The notebook installs everything else — `matplotlib`, `ipywidgets`, a CPU `torch`,
`nncf`, `onnx`, `einops`, `hydra-core`, `omegaconf`, `huggingface_hub` — into a notebook-local
`.venv`; inference itself needs only `numpy` and `openvino` plus the `fwam_ov.py` module and the
`utils/fwam_export/` exporter that sit next to the notebook.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/fasterWAM/README.md" />
