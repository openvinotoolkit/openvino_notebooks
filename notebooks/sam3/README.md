# SAM 3 Image Segmentation with OpenVINO

[SAM 3](https://github.com/facebookresearch/sam3) is Meta's unified vision-language model that supports:
- **Text-prompted segmentation**: Segment objects by describing them in natural language
- **Box-prompted segmentation**: Use bounding boxes as visual prompts (positive/negative)
- **Combined prompts**: Text + negative boxes to exclude regions
- **Point-prompted segmentation** (SAM1 task): Click foreground/background points
- **Video object tracking** (SAM2 task): Track objects across video frames

This notebook demonstrates how to convert SAM 3 models to OpenVINO IR format and run inference using OpenVINO,
aligned with the [official examples on HuggingFace](https://huggingface.co/facebook/sam3).

The model is decomposed into multiple sub-models for efficient conversion:
1. **Image Encoder** — ViT backbone with FPN neck
2. **Text Encoder** — Text transformer for language understanding
3. **Transformer Encoder** — Multi-level feature fusion with text cross-attention
4. **Transformer Decoder** — Object query decoding with box refinement
5. **Scoring** — Dot-product scoring + final box prediction
6. **Segmentation Head** — Pixel decoder + mask prediction
7. **Geometry Projections** — Box embedding projections (Linear/Conv2d/Embedding, no control flow)
8. **Geometry Cross-Attention** — CLS token + cross-attention encoder (no control flow)
9. **SAM1 Feature Prep** — conv_s0/s1 + no_mem_embed for SAM1 task features
10. **SAM2 Prompt Encoder** — Point/box prompt encoding (for SAM1/Tracker task)
11. **SAM2 Mask Decoder** — Mask prediction from prompts (for SAM1/Tracker task)


## Notebook Contents

This notebook shows an example of how to convert and use SAM3 using OpenVINO

Notebook contains the following steps:
1. Convert PyTorch models to OpenVINO format.
2. Run OpenVINO model in interactive segmentation mode.

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/sam3/README.md" />
