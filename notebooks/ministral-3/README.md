# Visual-language assistant with Ministral-3 and OpenVINO

Ministral-3 (Ministral-3-3B-Instruct-2512) is a lightweight, state-of-the-art multimodal model from Mistral AI, combining a 3.4B parameter language model with a 0.4B parameter vision encoder based on the Pixtral architecture. It is designed for efficient visual-language understanding tasks.

**Key Features of Ministral-3:**
* **Multimodal Understanding**: Combines text and vision capabilities in a compact 3B parameter model, enabling image understanding and visual question answering.
* **Long Context Support**: Supports up to 262,144 tokens with YaRN RoPE scaling for extended context processing.
* **Efficient Architecture**: Uses Grouped Query Attention (32 attention heads with 8 KV heads) for memory-efficient inference.
* **Pixtral Vision Encoder**: Employs a PixtralVisionModel with patch-based image processing and multi-modal projection for seamless vision-language integration.

More details about the model can be found in the [model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) and the [Mistral AI documentation](https://docs.mistral.ai/).

In this tutorial we consider how to convert and optimize Ministral-3 model for creating a multimodal chatbot using [Optimum Intel](https://github.com/huggingface/optimum-intel). Additionally, we demonstrate how to apply model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf).

## Notebook contents
The tutorial consists of the following steps:

- Install requirements
- Convert and Optimize model
- Prepare OpenVINO GenAI Inference Pipeline
- Run OpenVINO GenAI model inference
- Launch Interactive demo

In this demonstration, you'll create an interactive chatbot that can answer questions about provided image content.

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/ministral-3/README.md" />
