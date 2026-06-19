---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/LICENSE
language:
  - en
  - zh
library_name: openvino
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3-VL-2B-Instruct
base_model_relation: quantized
tags:
  - openvino
  - qwen3_vl
  - multimodal
  - vision
  - chat
  - conversational
---

# Qwen3-VL-2B-Instruct-int8-ov

* Model creator: [Qwen](https://huggingface.co/Qwen)
* Original model: [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

## Description

This is [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to INT8 by [NNCF](https://github.com/openvinotoolkit/nncf).

Qwen3-VL is the latest generation of vision-language models in the Qwen series, delivering comprehensive upgrades across the board: superior text understanding & generation, deeper visual perception & reasoning, extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities.

## Quantization Parameters

Weight compression was performed using `nncf.compress_weights` with the following parameters:

* mode: **INT8_ASYM**

For more information on quantization, check the [OpenVINO model optimization guide](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html).

## Compatibility

The provided OpenVINO™ IR model is compatible with:

* OpenVINO version 2026.1.0 and higher
* Optimum Intel 1.27.0 and higher

## Running Model Inference with [Optimum Intel](https://huggingface.co/docs/optimum/intel/index)

1. Install packages required for using [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) integration with the OpenVINO backend:

```
pip install -U "git+https://github.com/huggingface/optimum-intel.git" "openvino>=2026.1.0" "transformers>=4.57.0" "torch>=2.10.0" "pillow"
```

2. Run model inference:

```
import requests
from PIL import Image
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

model_id = "OpenVINO/Qwen3-VL-2B-Instruct-int8-ov"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = OVModelForVisualCausalLM.from_pretrained(model_id, trust_remote_code=True)

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

outputs = model.generate(**inputs, max_new_tokens=100)
generated = outputs[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(generated, skip_special_tokens=True)[0])
```

## Running Model Inference with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)

1. Install packages required for using OpenVINO GenAI.

```
pip install huggingface_hub pillow
pip install -U --pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly openvino openvino-tokenizers openvino-genai
```

2. Download model from HuggingFace Hub

```
import huggingface_hub as hf_hub

model_id = "OpenVINO/Qwen3-VL-2B-Instruct-int8-ov"
model_path = "Qwen3-VL-2B-Instruct-int8-ov"

hf_hub.snapshot_download(model_id, local_dir=model_path)
```

3. Run model inference:

```
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
import requests
from PIL import Image


def load_image(image_source):
    if isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
        image = Image.open(requests.get(image_source, stream=True).raw)
    else:
        image = Image.open(image_source)
    image_data = np.array(image.convert("RGB"))[None]
    return ov.Tensor(image_data)


device = "CPU"
pipe = ov_genai.VLMPipeline(model_path, device)

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image_tensor = load_image(image_url)
prompt = "Describe this image."

print(pipe.generate(prompt, image=image_tensor, max_new_tokens=100))
```

More GenAI usage examples can be found in OpenVINO GenAI library [docs](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md) and [samples](https://github.com/openvinotoolkit/openvino.genai?tab=readme-ov-file#openvino-genai-samples)

You can find more detailed usage examples in [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks):

* [Qwen3-VL multimodal chatbot](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen3-vl)

## Limitations

Check the original [model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) for limitations.

## Legal information

The original model is distributed under [Apache License 2.0](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/LICENSE) license. More details can be found in the [original model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
