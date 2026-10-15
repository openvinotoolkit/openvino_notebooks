# DeiT Tiny Image Classification with OpenVINO™

[DeiT (Data-efficient Image Transformers)](https://arxiv.org/abs/2012.12877) is a family of Vision Transformer models from Meta AI Research trained **without** any extra unlabelled data — only the 1.2 million images of ImageNet-1k are used, making it highly practical.

Unlike the original [ViT](https://arxiv.org/abs/2010.11929) which required hundreds of millions of images for pretraining, DeiT introduces a **teacher–student distillation token** and strong augmentation strategies (RandAugment, Mixup, CutMix, repeated augmentation) to train data-efficiently on ImageNet alone. A single 8-GPU node for 3 days is sufficient to train DeiT-base.

The model was proposed in [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) (2021) by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou from Meta AI Research.

You can find all DeiT checkpoints at [facebook on HuggingFace](https://huggingface.co/models?author=facebook&other=vit).

| Model | Params | ImageNet Top-1 | ImageNet Top-5 | HuggingFace |
|---|---|---|---|---|
| [DeiT-tiny](https://huggingface.co/facebook/deit-tiny-patch16-224) | **5 M** | 72.2% | 91.1% | `facebook/deit-tiny-patch16-224` |
| [DeiT-small](https://huggingface.co/facebook/deit-small-patch16-224) | 22 M | 79.9% | 95.0% | `facebook/deit-small-patch16-224` |
| [DeiT-base](https://huggingface.co/facebook/deit-base-patch16-224) | 86 M | 81.8% | 95.6% | `facebook/deit-base-patch16-224` |
| [DeiT-tiny distilled](https://huggingface.co/facebook/deit-tiny-distilled-patch16-224) | 6 M | 74.5% | 91.9% | `facebook/deit-tiny-distilled-patch16-224` |
| [DeiT-small distilled](https://huggingface.co/facebook/deit-small-distilled-patch16-224) | 22 M | 81.2% | 95.4% | `facebook/deit-small-distilled-patch16-224` |
| [DeiT-base distilled](https://huggingface.co/facebook/deit-base-distilled-patch16-224) | 87 M | 83.4% | 96.5% | `facebook/deit-base-distilled-patch16-224` |

This notebook defaults to **distilled DeiT Tiny** (`facebook/deit-tiny-distilled-patch16-224`, 5.9 M parameters). The distilled Small and Base variants are also available from the model selector.

---

## Preview

| Input image | OpenVINO FP16 prediction |
|---|---|
| ![LeBron James](assets/test_images/getty_LeBron.jpg) | `basketball` |

---

## What the Notebook Covers

- Load a distilled DeiT checkpoint via Hugging Face Transformers `AutoImageProcessor` and `AutoModelForImageClassification`
- Convert the PyTorch model to **OpenVINO IR** (FP16) using `ov.convert_model`
- Verify converted model inputs/outputs and inspect the IR
- Run image classification and compare PyTorch FP32, OpenVINO FP16, and OpenVINO INT8 predictions
- Evaluate **top-1 / top-5 accuracy** on a 300-image ImageNet-1k validation subset
- Quantize to **INT8** with NNCF accuracy-aware post-training quantization
- Benchmark FP16 and INT8 models on the OpenVINO device selected in the notebook

---

## Reference Results

Results vary by the selected model, device, and system. The checked-in notebook output uses distilled DeiT Tiny with OpenVINO `AUTO` and a 300-image ImageNet-1k validation subset.

| Runtime | Precision | Top-1 | Top-5 |
|---|---|---|---|
| PyTorch | FP32 | 76.67% | 94.00% |
| OpenVINO IR | FP16 | 76.67% | 94.00% |
| OpenVINO IR | INT8 | 75.67% | 94.00% |

The INT8 top-1 drop is 1.00 percentage point relative to the FP16 baseline, meeting the configured NNCF absolute accuracy-drop limit.

### IR File Sizes

| IR | BIN size | Compression |
|---|---|---|
| FP16 (`openvino_deit_tiny_distilled_patch16_224_fp16.xml`) | 11.8 MB | — |
| INT8 (`openvino_deit_tiny_distilled_patch16_224_int8.xml`) | 6.2 MB | ~1.9× vs FP16 |

---

## Installation Instructions

This is a self-contained example. We recommend running it in a dedicated virtual environment with Jupyter available.
For general environment setup, see the main [OpenVINO Notebooks Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md).

```bash
git clone --branch 2026.3 https://github.com/openvinotoolkit/openvino_notebooks.git
cd openvino_notebooks
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/deit-image-classification/deit_openvino.ipynb
```

Open `notebooks/deit-image-classification/deit_openvino.ipynb` and run all cells in order.

---

## References

- [Training data-efficient image transformers & distillation through attention (arXiv 2012.12877)](https://arxiv.org/abs/2012.12877)
- [facebookresearch/deit](https://github.com/facebookresearch/deit)
- [facebook/deit-tiny-distilled-patch16-224 on Hugging Face](https://huggingface.co/facebook/deit-tiny-distilled-patch16-224)
- [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino)
- [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

<img referrerpolicy="no-referrer-when-downgrade"
     src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/deit-image-classification/README.md" />
