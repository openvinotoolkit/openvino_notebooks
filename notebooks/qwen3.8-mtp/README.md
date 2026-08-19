# Qwen3.8 Multi-Token Prediction with OpenVINO

This experimental notebook demonstrates built-in Multi-Token Prediction (MTP) speculative decoding for Qwen3.8-27B with OpenVINO GenAI.

## Notebook

- [Qwen3.8 MTP with OpenVINO](./qwen3.8-mtp.ipynb)

## What the notebook covers

- Get the OpenVINO GenAI wheels built from PR #4065 for Windows or Linux, either by downloading the artifacts manually or automatically with a GitHub token, and install the ones matching the current environment.
- Get an MTP-enabled model, either from [OpenVINO/Qwen3.8-27B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov) or by exporting the original checkpoint with Optimum Intel PR #1814.
- Run image-and-text inference with `VLMPipeline`.
- Compare greedy baseline and MTP output, throughput, and acceptance metrics.
- Sweep the number of assistant tokens on a single prompt for a quick look at the trend.
- Sweep a grid of prompts and lookahead values that reports throughput, speedup, acceptance rate, and how many answers match the greedy baseline, and check that unsupported generation configurations are rejected.
- Cross-check performance against the `benchmark_vlm.py` sample, run once without and once with the draft model.

## Requirements

- A dedicated Python 3.10-3.13 environment.
- Windows x86_64 ([workflow run 31812000279](https://github.com/openvinotoolkit/openvino.genai/actions/runs/31812000279)) or Linux x86_64 with glibc 2.28 or newer, such as Ubuntu 20.04/22.04/24.04 ([workflow run 31812000248](https://github.com/openvinotoolkit/openvino.genai/actions/runs/31812000248)).
- Optionally, a GitHub token with Actions read access, taken from `GITHUB_TOKEN`, `GH_TOKEN`, `gh auth token`, or an interactive prompt. It is only needed for the automatic download; the manual path requires no token.
- Substantial memory and disk space for Qwen3.8-27B download or conversion and for inference.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen3.8-mtp/README.md" />
