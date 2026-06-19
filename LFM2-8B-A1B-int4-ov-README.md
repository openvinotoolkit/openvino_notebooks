---
license: other
license_name: lfm1.0
license_link: https://www.liquid.ai/lfm-license
library_name: openvino
pipeline_tag: text-generation
base_model: LiquidAI/LFM2-8B-A1B
tags:
  - openvino
  - lfm2
  - liquid
  - moe
  - edge
  - conversational
language:
  - en
  - de
  - fr
  - it
  - pt
  - hi
  - es
  - th
---

# LFM2-8B-A1B-int4-ov

* Model creator: [LiquidAI](https://huggingface.co/LiquidAI)
* Original model: [LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B)

> **EXPERIMENTAL MODEL**  
> This model has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

## Description

This is [LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to INT4 by [NNCF](https://github.com/openvinotoolkit/nncf).

LFM2-8B-A1B is a Mixture-of-Experts (MoE) language model with 8B total parameters and 1B active parameters per token, designed for efficient edge deployment.

## Quantization Parameters

Weight compression was performed using `nncf.compress_weights` with the following parameters:

* mode: **INT4_ASYM**
* group_size: **128**
* ratio: **0.8**

For more information on quantization, check the [OpenVINO model optimization guide](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html).

## Compatibility

The provided OpenVINO™ IR model is compatible with:

* OpenVINO version 2026.1.0 and higher
* Optimum Intel 1.27.0 and higher

## Running Model Inference with Optimum Intel

1. Install packages required for using Optimum Intel integration with the OpenVINO backend:

```bash
pip install git+https://github.com/huggingface/optimum-intel.git openvino
```

2. Run model inference:

```python
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

model_id = "OpenVINO/LFM2-8B-A1B-int4-ov"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = OVModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

inputs = tokenizer("What is a capital of France?", return_tensors="pt")
inputs.pop("token_type_ids", None)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

For more examples and possible optimizations, refer to the [Inference with Optimum Intel](https://huggingface.co/docs/optimum/intel/openvino/inference).

## Running Model Inference with OpenVINO GenAI

1. Install packages required for using OpenVINO GenAI.

```bash
pip install openvino-genai huggingface_hub
```

2. Download model from HuggingFace Hub

```python
import huggingface_hub as hf_hub

model_id = "OpenVINO/LFM2-8B-A1B-int4-ov"
model_path = "LFM2-8B-A1B-int4-ov"

hf_hub.snapshot_download(model_id, local_dir=model_path)
```

3. Run model inference:

```python
import openvino_genai as ov_genai

device = "CPU"
pipeline_config = {"ATTENTION_BACKEND": "SDPA"}
pipe = ov_genai.LLMPipeline(model_path, device, **pipeline_config)
print(pipe.generate("What is a capital of France?", max_length=200))
```

More GenAI usage examples can be found in [OpenVINO GenAI library docs](https://github.com/openvinotoolkit/openvino.genai/tree/master/src) and [samples](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples)

You can find more detailed usage examples in [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks):

* [LLM](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot)
* [RAG text generation](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-rag-langchain)

## Limitations

Check the original model card for [limitations](https://huggingface.co/LiquidAI/LFM2-8B-A1B).

## Legal information

The original model is distributed under [lfm1.0](https://www.liquid.ai/lfm-license) license. More details can be found in [LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel's Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). Intel's products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
