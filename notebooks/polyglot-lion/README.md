# Convert and Run Polyglot-Lion with OpenVINO

Convert the [Polyglot-Lion 1.7B checkpoint](https://huggingface.co/knoveleng/polyglot-lion-1.7b)
to a stateful OpenVINO model, compare the same audio against original PyTorch,
and explore weight compression on CPU and optional Intel GPU.

## Notebook Contents

[polyglot-lion.ipynb](./polyglot-lion.ipynb) follows the repository's prerequisites, model loading,
conversion, compression, accuracy validation, benchmarking, interactive demo and
conclusions sections. A single generated table of contents links to the sections;
short code cells keep each step separate from its explanation.

The workflow includes:

- pinned model/data download and original PyTorch CPU inference;
- stateful decoder export and a corrected full-utterance audio encoder;
- optional full INT8 weight compression or selective mixed INT8/FP16 weights;
- same-input transcripts, WER/CER, EOS checks and raw-logit diagnostics;
- complete-generation latency measurements; and
- an acknowledgement-gated sample/local-WAV demo.

The only model-specific runtime module is [polyglot_helper.py](./polyglot_helper.py).
The standard repository `notebook_utils.py` is copied from the checkout or
downloaded from a pinned revision. No research workspace, saved transcript,
report generator or local preconverted model is required.

## Installation Instructions

Follow the [installation guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/2026.3/README.md#-installation-guide),
select a dedicated Python 3.12 kernel with pip, and open this notebook from its
directory. Git is needed to install the pinned custom Optimum Intel revision.
Enable `INSTALL_PACKAGES` once, execute the setup cell, restart the kernel, then
disable the switch. Dependency pins are visible in that cell.

The validated environment is **Linux, Python 3.12**. Other platforms and Python
versions are not certified. CPU FP16 and one English clip are the defaults.
The first 1.7B export may exceed five minutes on slower machines. Allow about
16 GB for the checkpoint and all optional model variants, plus environment and
download caches. Conversion, download and compilation are not inference timings.

Options:

- `RUN_GPU=True`: add an available Intel GPU. The decoder explicitly requests f32
  before compilation (`GPU_DECODER_F32=True`); the encoder keeps its task settings.
- `RUN_MIXED_INT8=True`: keep the encoder FP16 and restore 136 decoder projection
  inputs to original FP16 after weight compression: all attention projections and
  all projections in layers 0-7. Remaining eligible decoder weights stay compressed.
- `RUN_INT8=True`: experimental full encoder/decoder weight compression. This
  configuration remains rejected by broader GPU accuracy testing.
- `RUN_MULTILINGUAL_EVALUATION=True`: use a longer English/Mandarin/Tamil/Malay
  FLEURS slice. This first-N slice is not the independent research acceptance set.
- `RUN_F32_DIAGNOSTICS=True`: separate global-f32 logit diagnostics, not a change
  to task evaluation, performance or deployment status.

The standard execution telemetry can be disabled by setting
`SCARF_NO_ANALYTICS=1` before launching Jupyter.

## Accuracy and precision limitations

The pinned PyTorch eager/SDPA encoder attends across the entire utterance after
removing CNN padding. The original exporter instead windowed attention and
included padding. The notebook explicitly fixes those semantics; Flash Attention
window behavior is not claimed equivalent. Full-context attention has quadratic
memory cost for long audio.

Corrected GPU inference still needs decoder-only f32 for the known Tamil 1666
control, even with FP16 weights. Neither default GPU decoder f16 nor blindly
forcing every compressed component to f32 is an accepted replacement. Full INT8
with decoder-only f32 fixes the three Tamil controls but fails a Mandarin gate
(+2.58 percentage points). It remains experimental.

The optional mixed first-eight-layer candidate passed the unchanged +2pp
per-language WER/CER and EOS gates versus PyTorch and corrected FP16 on the prior
256 clips, additional observed slices and a newly reserved 40-clip slice.
All three known Tamil regressions match PyTorch exactly.

| Slice | PyTorch Tamil WER | Mixed candidate Tamil WER |
|---|---:|---:|
| Prior 256 clips | 37.56% | 38.21% |
| New independent 40 clips | 30.52% | 30.52% |

These are historical, one-host research measurements, not recomputed by the
default smoke run. Identity/audio hashes were excluded across slices, not speakers
or globally translated sentence content. On the new 40, token sequences matched
35/40 overall and 7/10 Tamil: equal WER does not imply identical transcripts.
Original-model Tamil WER remains high.

The mixed BIN files were **26.10% smaller** than FP16. This is weight-only
compression, not activation PTQ or strict raw-logit parity.

Results stay report-only. A small smoke pass does not override known failures,
certify general multilingual robustness or approve deployment.

## Local performance and profiling findings

The accepted [DRO-Grasp PR 3766](https://github.com/openvinotoolkit/openvino_notebooks/pull/3766)
uses PyTorch CPU, not XPU. This tutorial follows that baseline; XPU enablement
and timing are not requirements.

Refreshed corrected-model measurements on Intel Core Ultra X7 358H / Arc B390
(PTL), Linux, OpenVINO 2026.4.1, PyTorch 2.15.0+cpu, 16 PyTorch threads:
one warmup and three complete-generation measurements, median seconds below.
Runs were sequential on a shared host, not isolated hardware certification.

| Runtime | English | Mandarin | Tamil 1666 | Malay |
|---|---:|---:|---:|---:|
| PyTorch CPU FP32 | 2.620 | 1.992 | 15.929 | 3.498 |
| OpenVINO FP16 CPU | 2.859 | 2.458 | 12.646 | 2.976 |
| OpenVINO FP16 GPU, decoder f32 | 1.885 | 1.277 | 14.515 | 2.721 |
| OpenVINO mixed CPU | 1.923 | 1.782 | 9.225 | 2.118 |
| OpenVINO mixed GPU, decoder f32 | 1.961 | 1.340 | 15.187 | 2.869 |

These are four clips (IDs 1904/1906/1666/1697), not corpus throughput. Timings
include encoder, prefill, all decode steps, Python orchestration and transfers;
exclude load/compile, audio loading, preprocessing and detokenization. All 20
cases reached EOS, repeated tokens identically within each case and shared the
same respective transcript and output length (24/16/181/36) across runtimes.
Mixed weights improved these CPU timings but showed **no GPU latency speedup**.
The CPU timing clips do not establish broad CPU mixed-weight accuracy acceptance.

GPU generation profiling found reference-named kernels outside the top five,
with 2.329% FP16 / 2.236% mixed aggregate executed-node time. Decode `lm_head`
was largest (14.28% / 14.97%), using f32 GEMM even with compressed weights.
Kernel-time shares are not wall-time shares. All 28 gate-projection/Swish/Multiply
patterns were already fused. Distinct Q/K/V projections sharing normalized
inputs and separate attention Gemm/SoftMax paths are investigation opportunities,
not implemented fusions or promised speedups.

The requested roofline tool at `5eca2d8c1260015398bb5c26093ddd2a3fd984d2`
measured 131.062 GB/s bandwidth. With actual f32 decoder/KV, it estimates
compute-bound prefill and memory-bound first decode. FP16/mixed first-decode
medians were 74.158/76.849 ms; estimated ideal times were 27.106/18.981 ms.
Decoder shape inference retained 79 prefill / 100 decode warnings. The
whole-utterance encoder's measured 11.34 ms was slightly below its 11.73 ms
estimate, illustrating that these are heuristic estimates, not hard physical
bounds or efficiency certificates.

Repeated prefill-only raw-IR microbenchmarks were rejected after unstable
last-token logits. Final phase timings replayed prefill then first decode before
reset, with the completion decode outside prefill timing. All decoder trials
matched actual-generation top-1 and measured repeat-logit error stayed below
`1e-3` (maximum `0.000484`). This protocol check is not a PyTorch logit-parity
certificate or proof of a specific plugin defect.

These are static local research findings, not extra default notebook steps.
The tested platform scope is this CPU and GPU only; NPU and other hardware
remain untested. Profiling and roofline tooling are not runtime dependencies.

## Sources and licenses

- [Polyglot-Lion source](https://github.com/knoveleng/polyglot-lion) and
  [checkpoint](https://huggingface.co/knoveleng/polyglot-lion-1.7b): MIT.
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR): Apache-2.0; the corrected
  encoder follows its implementation.
- [Pinned Optimum Intel](https://github.com/openvino-dev-samples/optimum-intel/tree/4ca1144eafc3ef7d3d805a99c7b92953441437e5):
  Apache-2.0.
- [FLEURS](https://huggingface.co/datasets/google/fleurs): CC-BY-4.0,
  Google / Conneau et al. (2022), [paper](https://arxiv.org/abs/2205.12446).

Model/data revisions and audio hashes are recorded in the generated results.
Preserve upstream license/NOTICE/attribution requirements; see
[third-party notices](../../licensing/third-party-programs.txt).

## Cleanup

After shutting down the kernel, remove only this tutorial's generated `model/`,
`data/`, `output/` and downloaded `notebook_utils.py` if no longer needed.
They are ignored by Git. Do not remove shared model caches.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/polyglot-lion/README.md" />
