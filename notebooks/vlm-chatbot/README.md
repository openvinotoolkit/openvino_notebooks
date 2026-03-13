# Create a VLM Chatbot with OpenVINO

This notebook shows how to build a Vision-Language Model (VLM) chatbot with OpenVINO. A VLM can process both text and images, which makes it useful for multimodal chat, image understanding, visual question answering, and grounded reasoning over image inputs.

The example is focused on the OpenVINO Generate API workflow and demonstrates how to prepare a supported VLM, convert or download OpenVINO models, and run interactive multimodal inference.

Check out the demo
![demo](https://github.com/user-attachments/assets/3435c352-5d27-40ac-b07f-47ab9c43432a)


## Notebook

- [VLM chatbot with Generate API](./vlm-chatbot-generate-api.ipynb)

## Supported VLM Models

The notebook exposes the VLM models currently defined in `SUPPORTED_VLM_MODELS` in `utils/llm_config.py`.

### English

- `Llava-Next-Video-7B` - Hugging Face model: [`llava-hf/LLaVA-NeXT-Video-7B-hf`](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf)
- `Qwen3-Vl-8B-Instruct` - Hugging Face model: [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- `Qwen2.5-VL-3B-Instruct` - Hugging Face model: [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

### Chinese

- `Qwen3-VL-8B-Instruct` - Hugging Face model: [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- `Qwen2.5-VL-3B-Instruct` - Hugging Face model: [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

### Japanese

- `Qwen3-VL-8B-Instruct` - Hugging Face model: [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- `Qwen2.5-VL-3B-Instruct` - Hugging Face model: [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## Notes

- All currently supported VLM entries are marked as unavailable on `NPU` in the configuration.
- Conversion of larger multimodal models can require substantial system memory and time.
- Some Hugging Face models may require accepting the model license before download.

## What The Notebook Covers

- Install notebook prerequisites
- Select a supported VLM and target precision
- Download or convert the model to OpenVINO format
- Build a multimodal chat pipeline with OpenVINO Generate API
- Run image-plus-text inference in an interactive chatbot flow

## Installation Instructions

This is a self-contained example that relies on the notebook-local helper code. We recommend running it in a dedicated virtual environment with Jupyter available.

For general environment setup, see the main [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/vlm-chatbot/README.md" />
