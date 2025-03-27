# Optical Character Recognition with GOT-OCR 2.0 and OpenVINO
 GOT-OCR2.0 is unified End-to-End model for recognition text on images.

![](https://github.com/user-attachments/assets/85b706e1-3179-4de2-bbb7-945c23d02678)

The GOT-OCR 2.0 model was introduced in the paper: [**General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model**](https://arxiv.org/abs/2409.01704)

### Key Features
GOT-OCR 2.0 is a **state-of-the-art OCR model** designed to handle a wide variety of tasks, including:
- **Plain Text OCR**
- **Formatted Text OCR**
- **Fine-grained OCR**
- **Multi-crop OCR**
- **Multi-page OCR**
### Beyond Text
GOT-OCR 2.0 has also been fine-tuned to work with non-textual data, such as:
- **Charts and Tables**
- **Math and Molecular Formulas**
- **Geometric Shapes**
- **Sheet Music**
  
In this tutorial we consider how to convert and run GOT-OCR 2.0 model using OpenVINO [Optimum Intel](https://github.com/huggingface/optimum-intel). Additionally, we demonstrate how to apply model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf).

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
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/got-ocr2/README.md" />
