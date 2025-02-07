# Visual-language assistant with Qwen2VL and OpenVINO

Qwen2VL is the latest addition to the QwenVL series of multimodal large language models.

**Key Enhancements of Qwen2VL:**
* **SoTA understanding of images of various resolution & ratio**: Qwen2-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.
* **Understanding videos of 20min+**: Qwen2-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.
* **Agent that can operate your mobiles, robots, etc.:** with the abilities of complex reasoning and decision making, Qwen2-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.
* **Multilingual Support:** to serve global users, besides English and Chinese, Qwen2-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.


**Model Architecture Details:**

* **Naive Dynamic Resolution**: Qwen2-VL can handle arbitrary image resolutions, mapping them into a dynamic number of visual tokens, offering a more human-like visual processing experience.

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/qwen2_vl.jpg" width="50%"/>
<p>

* **Multimodal Rotary Position Embedding (M-ROPE)**: Decomposes positional embedding into parts to capture 1D textual, 2D visual, and 3D video positional information, enhancing its multimodal processing capabilities.

<p align="center">
    <img src="http://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/mrope.png" width="50%"/>
<p>



More details about model can be found in [model card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [blog](https://qwenlm.github.io/blog/qwen2-vl/) and original [repo](https://github.com/QwenLM/Qwen2-VL).

In this tutorial we consider how to convert and optimize Qwen2VL model for creating multimodal chatbot using [Optimum Intel](https://github.com/huggingface/optimum-intel). Additionally, we demonstrate how to apply model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf)

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Prepare OpenVINO GenAI Inference Pipeline
- Run OpenVINO GenAI model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.

The image bellow illustrates example of input prompt and model answer.
![example.png](https://github.com/user-attachments/assets/7e12ac6c-12f8-43d8-9c0a-b63d6ecaf20b)

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen2-vl/README.md" />
