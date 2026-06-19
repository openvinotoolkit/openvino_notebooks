---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model:
  - Qwen/Qwen3.5-35B-A3B
base_model_relation: quantized


---
# Qwen3.5-35B-A3B-int4-ov
 * Model creator: [Qwen](https://huggingface.co/Qwen)
 * Original model: [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

## Description
This is [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to INT4 by [NNCF](https://github.com/openvinotoolkit/nncf).

## Quantization Parameters

Weight compression was performed using `nncf.compress_weights` with the following parameters:

 * mode: **INT4_ASYM**
 * ratio: **1.0**
 * group_size: **128**
 * backup_mode: **INT8_ASYM**
 * ignored_scope: layers matching `.*shared_expert.*` and `.*attn.*` are kept in the backup precision (INT8_ASYM)

For more information on quantization, check the [OpenVINO model optimization guide](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html).

## Compatibility

The provided OpenVINO™ IR model is compatible with:

* OpenVINO version 2026.2.0 and higher
* Optimum Intel 1.27.0 and higher

## Running Model Inference with [Optimum Intel](https://huggingface.co/docs/optimum/intel/index)

1. Install packages required for using [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) integration with the OpenVINO backend:

```
pip install optimum[openvino]
```

2. Run model inference:

```
import requests
from PIL import Image
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

model_id = "OpenVINO/Qwen3.5-35B-A3B-int4-ov"
processor = AutoProcessor.from_pretrained(model_id)
model = OVModelForVisualCausalLM.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")

# Temporary workaround for optimum-intel (main): `mm_token_type_ids` is required by
# `_prepare_position_ids_for_generation` / `get_rope_index`, but is not declared in the OV model's
# forward signature, so `transformers.GenerationMixin._validate_model_kwargs` would reject it.
# Make the validator ignore this key while keeping it in the real `model_kwargs`.
_original_validate = model._validate_model_kwargs
model._validate_model_kwargs = lambda kw: _original_validate(
    {k: v for k, v in kw.items() if k != "mm_token_type_ids"}
)

outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])
```

For more examples and possible optimizations, refer to the [Inference with Optimum Intel](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html).

## Running Model Inference with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)


1. Install packages required for using OpenVINO GenAI.
```
pip install huggingface_hub "Pillow"
pip install --pre -U openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

2. Download model from HuggingFace Hub
   
```
import huggingface_hub as hf_hub

model_id = "OpenVINO/Qwen3.5-35B-A3B-int4-ov"
model_path = "Qwen3.5-35B-A3B-int4-ov"

hf_hub.snapshot_download(model_id, local_dir=model_path)

```

3. Run model inference:

```
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
import requests
from PIL import Image

device = "CPU"
pipe = ov_genai.VLMPipeline(model_path, device)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image_tensor = ov.Tensor(np.array(image)[None])

print(pipe.generate("Describe this image.", image=image_tensor, max_new_tokens=200))
```

More GenAI usage examples can be found in OpenVINO GenAI library [docs](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html) and [samples](https://github.com/openvinotoolkit/openvino.genai?tab=readme-ov-file#openvino-genai-samples)

You can find more detaild usage examples in OpenVINO Notebooks:

- [Qwen3-VL multimodal chatbot](https://openvinotoolkit.github.io/openvino_notebooks/?search=qwen3-vl)
- [Visual-language assistant](https://openvinotoolkit.github.io/openvino_notebooks/?tasks=Image-to-Text)

## Limitations

Check the original [model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) for limitations.

## Legal information

The original model is distributed under [Apache License Version 2.0](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/LICENSE) license. More details can be found in [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
