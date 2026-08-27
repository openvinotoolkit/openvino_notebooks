# Speaker Diarization Enablement (PyTorch + OpenVINO)

This folder enables and benchmarks the pyannote speaker-diarization pipeline
(`pyannote/speaker-diarization-community-1` with `pyannote/segmentation-3.0`)
across CPU, Intel XPU, and OpenVINO (CPU/GPU) backends on the VoxConverse dataset.

## What was enabled

- **PyTorch CPU** — baseline diarization on CPU (`diar_cpu`).
- **PyTorch XPU** — diarization on Intel GPU via the XPU PyTorch build (`diar_xpu`).
- **OpenVINO CPU** — diarization using exported OpenVINO IR on CPU (`diar_ov`).
- **OpenVINO GPU** — same OpenVINO IR running on the Intel iGPU (`diar_ov`), in **FP16**.

The two heavy neural blocks (segmentation and speaker embedding) are exported to
OpenVINO IR (`.xml` / `.bin`); the runtime picks a dynamic/static shape strategy
automatically per device.

## Precision

- The exported IR weights are stored in **FP16** (`ov.save_model` default `compress_to_fp16=True`).
- On **GPU**, inference runs in **FP16** (OpenVINO's default inference precision for Intel GPUs).
- On **CPU**, inference runs in FP32 (weights are decompressed at load time).

## Accuracy

- Full VoxConverse (test) DER: originally committed **11.2%**, currently observed **8.3%**.

## What this provides

- One-command diarization on a single file per backend.
- Short 80–100 second per-file latency benchmarks.
- Full VoxConverse DER (Diarization Error Rate) scoring per backend.
- A consolidated notebook (`inference.ipynb`) that runs the full workflow:
  environment setup, Hugging Face access, dataset download, IR export, smoke
  tests, benchmarks, and DER scoring.

## Key files

| File | Purpose |
|---|---|
| `inference.ipynb` | End-to-end notebook covering all backends |
| `BKM_VOXCONVERSE_DIARIZATION.md` | Step-by-step runbook |
| `export_pyann.py` | Exports the diarization models to OpenVINO IR |
| `run_diarization.py` | PyTorch CPU/XPU single-file diarization |
| `run_diarization_ov.py` | OpenVINO CPU/GPU single-file diarization |
| `run_file_benchmark.sh` | Per-file latency benchmark helper |
| `score_der.py` | DER scoring on Debug or VoxConverse |
| `diar_cpu.yaml` / `diar_xpu.yaml` / `diar_ov.yaml` | Conda environments per backend |

## Environments

| Backend | Conda env |
|---|---|
| `cpu` | `diar_cpu` |
| `xpu` | `diar_xpu` |
| `ov-cpu`, `ov-gpu` | `diar_ov` |

See `BKM_VOXCONVERSE_DIARIZATION.md` for the full setup and run instructions.
