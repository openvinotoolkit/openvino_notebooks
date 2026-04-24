# Multilingual Text Embedding on Intel® NPU with OpenVINO™

This notebook demonstrates how to run the [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
sentence-embedding model on the Intel® NPU (Core Ultra series) using OpenVINO™.

The notebook focuses on a quirk that often blocks first-time NPU users:
**dynamic input shapes are not supported by the NPU compiler**. The notebook
walks through exporting the model to OpenVINO IR, applying a static shape
(`[1, 512]`), compiling for the NPU, and using it as a multilingual semantic
search backend with an in-memory FAISS index.

## Notebook Contents

The tutorial consists of the following steps:

1. Install the prerequisites
2. Export `intfloat/multilingual-e5-small` to OpenVINO IR with `optimum-intel`
3. Reshape the IR to a static `[1, 512]` shape required by the NPU
4. Compile on NPU (with CPU fallback) and benchmark embedding latency
5. Build a small in-memory FAISS index and run a multilingual retrieval demo

The notebook uses [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(117 M parameters, 100+ languages, 384-dim output), which is small enough for
the NPU and strong enough for retrieval. A pre-exported NPU-ready IR is also
available at
[`seminse/multilingual-e5-small-openvino-npu-static`](https://huggingface.co/seminse/multilingual-e5-small-openvino-npu-static)
if you would like to skip the export step.

### Why a static shape?

The Intel NPU compiler (NPU 3720 and newer) rejects dynamic input dims (`-1`
in the shape) and fails with `Got negative shape dim bound: '-1'`. The default
`optimum-intel` export keeps shapes dynamic, which works fine on CPU/GPU but
not on the NPU. The notebook applies `model.reshape({input_name: [1, 512]})`
before compilation so the same IR runs on NPU, GPU, and CPU.

## Installation Instructions

This is a self-contained example that relies solely on its own code.<br/>
We recommend running the notebook in a virtual environment. You only need a
Jupyter server to start. For details, please refer to
[Installation Guide](../../README.md).

The notebook gracefully falls back to CPU if no NPU is available, so it can
be executed on any machine that has OpenVINO installed.
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/npu-multilingual-embedding/README.md" />
