# Multilingual Text Embedding on Intel® NPU with OpenVINO™

This notebook demonstrates how to run the [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
sentence-embedding model with OpenVINO™ across **every device a Core Ultra
laptop exposes** — CPU, integrated GPU (Arc), and NPU (3720) — using a
single static-shape OpenVINO IR. The notebook compiles the same `.xml` /
`.bin` for whichever device is available and runs end-to-end on a regular
laptop without special hardware.

The notebook focuses on a quirk that often blocks first-time NPU users:
**dynamic input shapes are not supported by the NPU compiler**. The
notebook walks through exporting the model to OpenVINO IR, applying a
static shape (`[1, 512]`), compiling for the best available device, and
using it as a multilingual semantic search backend with an in-memory
FAISS index.

## Notebook Contents

The tutorial consists of the following steps:

1. Install the prerequisites
2. Export `intfloat/multilingual-e5-small` to OpenVINO IR with `optimum-intel`
3. Reshape the IR to a static `[1, 512]` shape required by the NPU
4. Compile on the best available device (NPU → GPU → CPU) and benchmark embedding latency
5. Build a small in-memory FAISS index and run a multilingual retrieval demo

The notebook uses [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(117 M parameters, 100+ languages, 384-dim output), which is small enough
for the NPU and strong enough for retrieval. A pre-exported NPU-ready IR
is also available at
[`seminse/multilingual-e5-small-openvino-npu-static`](https://huggingface.co/seminse/multilingual-e5-small-openvino-npu-static)
if you would like to skip the export step.

### Why a static shape?

The Intel NPU compiler (NPU 3720 and newer) rejects dynamic input dims
(`-1` in the shape) and fails with `Got negative shape dim bound: '-1'`.
The default `optimum-intel` export keeps shapes dynamic, which works fine
on CPU/GPU but not on NPU. The notebook applies
`model.reshape({input_name: [1, 512]})` before compilation so **the same
IR runs on NPU, GPU, and CPU**.

### Measured on a real laptop (Core Ultra 9, OpenVINO 2026.1.0)

Same `openvino_model.xml`, mean of 50 runs after warm-up:

| Device | Latency / text | Throughput | Compile | Notes |
|---|---:|---:|---:|---|
| **GPU.0** (Intel Arc iGPU) | 19.4 ms | 51.6 /s | 4.2 s | Fastest raw speed. |
| **NPU** (3720) | 29.1 ms | 34.3 /s | 0.5 s | ~1 W — leaves CPU + GPU free for other workloads. |
| **CPU** | 49.0 ms | 20.4 /s | 0.4 s | Universal fallback; runs anywhere. |

You do **not** need an NPU laptop to run this notebook. The same code
path works on any machine with OpenVINO; the device just changes.

The headline use case for the NPU specifically is keeping the embedder
**always-on at ~1 W while the CPU and GPU run other workloads**
concurrently — for example, a local LLM served via `llama.cpp` on a
discrete GPU. In that setup the NPU is the only otherwise-idle piece of
silicon, so pinning embeddings there gives a memory layer that doesn't
contend with the LLM.

## Installation Instructions

This is a self-contained example that relies solely on its own code.<br/>
We recommend running the notebook in a virtual environment. You only need
a Jupyter server to start. For details, please refer to
[Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/npu-multilingual-embedding/README.md" />
