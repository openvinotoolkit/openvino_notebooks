# Qwen3.8 Multi-Token Prediction with OpenVINO

This experimental notebook demonstrates built-in Multi-Token Prediction (MTP) speculative decoding for Qwen3.8-27B with OpenVINO GenAI.

## Notebook

- [Qwen3.8 MTP with OpenVINO](./qwen3.8-mtp.ipynb)

## What the notebook covers

- Install OpenVINO GenAI from the nightly index and Optimum Intel from PR #1814.
- Download a preconverted MTP-enabled [INT4](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov) or [INT8](https://huggingface.co/OpenVINO/Qwen3.8-27B-int8-ov) model, or explicitly select local export of the original checkpoint.
- Run image-and-text inference with `VLMPipeline`.
- Compare greedy baseline and MTP output, throughput, and acceptance metrics.
- Sweep the number of assistant tokens on a single prompt for a quick look at the trend.
- Sweep a grid of prompts and lookahead values that reports throughput, speedup, acceptance rate, and how many answers match the greedy baseline, and check that unsupported generation configurations are rejected.
- Cross-check performance against the `benchmark_vlm.py` sample, run once without and once with the draft model.

## Requirements

- A dedicated Python environment.
- Substantial memory and disk space for Qwen3.8-27B download or conversion and for inference.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen3.8-mtp/README.md" />
