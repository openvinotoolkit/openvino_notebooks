# PDF converting with olmOCR model and OpenVINO

PDF documents offer a vast source of high-quality tokens for training language models, but their diverse formats and layouts present challenges for content extraction. olmOCR is introduced as an open-source Python toolkit that processes PDFs into clean, linearized plain text while preserving structured content like sections, tables, lists, and equations. The toolkit utilizes a fine-tuned 7B vision language model trained on a diverse sample of 260,000 pages from over 100,000 PDFs. olmOCR is optimized for large-scale batch processing and includes all components such as VLM weights, data, training code, and inference code.

**Model Capabilities**:
* olmOCR is capable of processing a diversity of document types, covering different domains as well as visual layouts. It uses Markdown to represent structured content, such as sections, lists, equations and tables.
* olmOCR uses both text and visual information to obtain an accurate text representation of a documents. The authors develop document-anchoring, a technique to extract text and layout information from born-digital PDF documents. document-anchoring can be used to prompt VLMs alongside images of document pages to significantly improve extraction
* To build olmOCR, the authors curate olmOCR-mix-0225, a dataset of nearly 260,000 PDF pages from a diverse set of PDFs crawled from the web and public domain books. This corpus is used to fine-tune olmOCR-7B-0225-preview from Qwen2-VL-7B-Instruct. olmOCR-mix-0225 is released to facilitate further research in document extraction, and open source model weights and code as part of the toolkit.


![image](https://github.com/user-attachments/assets/2bc83cd4-894f-464f-8fd6-0ac9c9105c80)
  
In this tutorial we consider how to convert and run olmOCR-7B-0225-preview model using OpenVINO [Optimum Intel](https://github.com/huggingface/optimum-intel). Additionally, we demonstrate how to apply model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf).

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/olmocr-pdf-vlm/README.md" />
