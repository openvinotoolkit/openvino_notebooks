# Hunyuan Machine Translation with OpenVINO

This notebook supports two series of machine translation models developed by Tencent:

**[HY-MT1.5](https://huggingface.co/tencent/HY-MT1.5-1.8B)** — Built on the Hunyuan Dense V1 architecture, delivers high-quality translation across **33+ languages**, including Chinese, English, Japanese, Korean, French, German, Spanish, Arabic, and many more.
- **[HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B)** — Lightweight model suitable for resource-constrained environments
- **[HY-MT1.5-7B](https://huggingface.co/tencent/HY-MT1.5-7B)** — Larger model with higher translation quality

For more details about HY-MT1.5, please refer to the [technical report](https://arxiv.org/abs/2512.24092).

**[Hy-MT2](https://huggingface.co/tencent/Hy-MT2-7B)** — The next-generation series with improved translation quality and broader language coverage.
- **[Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)** — Lightweight next-gen model
- **[Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B)** — Higher quality next-gen model

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Select the model (HY-MT1.5-1.8B, HY-MT1.5-7B, Hy-MT2-1.8B, or Hy-MT2-7B) and weight compression format (INT4/INT8/FP16)
- Download and convert the model to OpenVINO IR format using [Optimum Intel](https://huggingface.co/docs/optimum/intel/index)
- Compress model weights using [NNCF](https://github.com/openvinotoolkit/nncf)
- Create a translation inference pipeline with [OpenVINO Generate API](https://github.com/openvinotoolkit/openvino.genai)
- Run an interactive Gradio translation demo with 33+ language support

## Supported Languages

Chinese, English, French, Portuguese, Spanish, Japanese, Turkish, Russian, Arabic, Korean, Thai, Italian, German, Vietnamese, Malay, Indonesian, Filipino, Hindi, Traditional Chinese, Polish, Czech, Dutch, Khmer, Burmese, Persian, Gujarati, Urdu, Telugu, Marathi, Hebrew, Bengali, Tamil, Ukrainian, and more.

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

## Installation Instructions
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/hunyuan-translation/README.md" />
