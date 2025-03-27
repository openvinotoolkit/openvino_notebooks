# Visual-language assistant with Qwen2VL and OpenVINO

 Qwen2.5-VL is the latest addition to the QwenVL series of multimodal large language models.

 ![](https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2.5-vl-Capybara.png)

**Key Enhancements of Qwen2.5VL:**
* **Understand things visually**: Qwen2.5-VL is not only proficient in recognizing common objects such as flowers, birds, fish, and insects, but it is highly capable of analyzing texts, charts, icons, graphics, and layouts within images.
* **Being agentic**: Qwen2.5-VL directly plays as a visual agent that can reason and dynamically direct tools, which is capable of computer use and phone use.
* **Understanding long videos and capturing events**: Qwen2.5-VL can comprehend videos of over 1 hour, and this time it has a new ability of capturing event by pinpointing the relevant video segments
* **Capable of visual localization in different formats**: Qwen2.5-VL can accurately localize objects in an image by generating bounding boxes or points, and it can provide stable JSON outputs for coordinates and attributes.
* **Generating structured outputs**: for data like scans of invoices, forms, tables, etc. Qwen2.5-VL supports structured outputs of their contents, benefiting usages in finance, commerce, etc.

**Model Capabilities**:
* **World-wide Image Recognition**. Qwen2.5-VL has significantly enhanced its general image recognition capabilities, expanding the categories of images to an ultra-large number. It not only includes plants, animals, landmarks of famous mountains and rivers, but also IPs from film and TV series, as well as a wide variety of products.
* **Precise Object Grounding**. Qwen2.5-VL utilizes bounding boxes and point-based representations for grounding, enabling hierarchical positioning and standardized JSON output. This enhanced localization capability serves as a foundation for visual reasoning.
* **Enhanced Text Recognition and Understanding**. Qwen2.5-VL has upgraded its OCR recognition capabilities to a new level, with enhanced multi-scenario, multi-language and multi-orientation text recognition and text localization performance. Furthermore, it has been significantly enhanced in information extraction to meet the growing digitalized and intelligent demands in areas such as qualification review and financial business.
* **Powerful Document Parsing**. Qwen2.5-VL has designed a unique document parsing format called QwenVL HTML format, which extracts layout information based on HTML. QwenVL HTML can perform document parsing in various scenarios, such as magazines, research papers, web pages, and even mobile screenshots.
* **Enhanced Video Comprehension Ability**. Qwen2.5-VL video comprehension capabilities have been comprehensively upgraded. In terms of temporal processing, we have introduced dynamic frame rate (FPS) training and absolute time encoding technology. As a result, the model can not only support the understanding of ultra-long videos on an hourly scale but also achieve second-level event localization. It is capable of accurately comprehending content from long videos spanning hours, searching for specific events within videos, and summarizing key points from different time segments. This allows users to quickly and efficiently extract crucial information embedded in the videos.

**Model Architecture Details:**

Comparing with Qwen2VL, Qwen2.5VL architecture receives following updates:
* **Dynamic Resolution and Frame Rate Training for Video Understanding**
Extended dynamic resolution to the temporal dimension by adopting dynamic FPS sampling, enabling the model to comprehend videos at various sampling rates. Accordingly, mRoPE was updated in the time dimension with IDs and absolute time alignment, enabling the model to learn temporal sequence and speed, and ultimately acquire the ability to pinpoint specific moments.
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-VL/qwen2.5vl_arc.jpeg)
* **Streamlined and Efficient Vision Encoder**
Qwen2.5VL enhances both training and inference speeds by strategically implementing window attention into the ViT. The ViT architecture is further optimized with SwiGLU and RMSNorm, aligning it with the structure of the Qwen2.5 LLM.

More details about model can be found in [model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [blog](https://qwenlm.github.io/blog/qwen2.5-vl/), [technical report](https://arxiv.org/abs/2502.13923) and original [repo](https://github.com/QwenLM/Qwen2.5-VL).

In this tutorial we consider how to convert and optimize Qwen2.5VL model for creating multimodal chatbot using [Optimum Intel](https://github.com/huggingface/optimum-intel). Additionally, we demonstrate how to apply model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.

The image bellow illustrates example of input prompt and model answer.
![example.png](https://github.com/user-attachments/assets/7e12ac6c-12f8-43d8-9c0a-b63d6ecaf20b)

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen2.5-vl/README.md" />
