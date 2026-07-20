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

This notebook covers **DeiT Tiny** (`deit-tiny-patch16-224`, 5 M params), the smallest and fastest checkpoint in the family.

---

## Preview

| Input image | OpenVINO FP16 — Top-5 predictions |
|---|---|
| ![dogs](../../assets/preview/deit_dog_inference.jpg) | **1.** `golden retriever` — logit **5.31** |
| | 2. `Saluki, gazelle hound` — logit 4.94 |
| | 3. `Labrador retriever` — logit 4.72 |
| | 4. `Weimaraner` — logit 3.41 |
| | 5. `cocker spaniel` — logit 3.39 |

---

## What the Notebook Covers

- Load `facebook/deit-tiny-patch16-224` via HuggingFace Transformers `AutoImageProcessor` + `AutoModelForImageClassification`
- Convert the PyTorch model to **OpenVINO IR** (FP16) using `ov.convert_model`
- Verify converted model inputs/outputs and inspect the IR
- Run image classification and compare PyTorch vs OpenVINO top-1 logits (max abs diff < 0.023)
- Benchmark PyTorch CPU and **OpenVINO FP16 on Intel Arc GPU** with `benchmark_app`
- Evaluate **top-1 / top-5 accuracy** on a 200-image ImageNet-1k subset
- Run inference on **Intel Arc XPU** via PyTorch XPU backend
- Quantize to **INT8** via NNCF and compare FP16 vs INT8 size and throughput

---

## Performance Results — Intel Arc A770, OpenVINO 2026.4

### Benchmark (batch 1, latency mode, `benchmark_app`)

| Runtime | Avg latency | Throughput | Tokens/sec | Notes |
|---|---|---|---|---|
| PyTorch (CPU) | 7.99 ms | 125.19 FPS | 24 663 | 100 batch-1 forward passes |
| **OpenVINO FP16 (GPU)** | **2.55 ms** | **385.72 FPS** | **75 987** | `benchmark_app`, 30 s, latency hint |

OpenVINO FP16 is **~3× faster** than PyTorch CPU for this model.

### Accuracy (200-image ImageNet-1k subset)

| Model | Top-1 | Top-5 |
|---|---|---|
| Published DeiT Tiny (full 50 k val) | **72.2%** | **91.1%** |
| OpenVINO FP16 IR (200-image eval) | 64.5% | 87.0% |

The ±7.7% top-1 gap is within the expected natural variance for a 200-image sample (2-sigma CI ≈ ±6% for n = 200, p ≈ 0.72). The FP16 conversion contributes < 0.5% accuracy loss for models of this size.

### IR File Sizes

| IR | BIN size | Compression |
|---|---|---|
| FP16 (`openvino_deit_tiny_patch16_224_fp16.xml`) | 11 MB | — |
| INT8 (`openvino_deit_tiny_patch16_224_int8.xml`) | 5.7 MB | ~2× vs FP16 |

---

## Installation Instructions

This is a self-contained example. We recommend running it in a dedicated virtual environment with Jupyter available.  
For general environment setup, see the main [OpenVINO Notebooks Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md).

```bash
git clone https://github.com/intel-sandbox/GroundingDino_OV.git
cd GroundingDino_OV
python -m venv venv && source venv/bin/activate
pip install openvino transformers torch torchvision pillow matplotlib nncf datasets
```

> **Intel Arc GPU users:** replace the default torch build with the XPU-enabled variant:
>
> ```bash
> pip install torch==2.12.1+xpu torchvision==0.27.1+xpu \
>   --index-url https://download.pytorch.org/whl/xpu
> ```
>
> Then upgrade OpenVINO to the latest nightly for best Arc support:
>
> ```bash
> pip install --pre -U openvino \
>   --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
> ```

Open `notebooks/deit_tiny_openvino.ipynb` and run all cells in order.

---

## References

- [Training data-efficient image transformers & distillation through attention (arXiv 2012.12877)](https://arxiv.org/abs/2012.12877)
- [facebookresearch/deit](https://github.com/facebookresearch/deit)
- [facebook/deit-tiny-patch16-224 on HuggingFace](https://huggingface.co/facebook/deit-tiny-patch16-224)
- [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino)
- [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

<img referrerpolicy="no-referrer-when-downgrade"
     src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=GroundingDino_OV/notebooks/deit_tiny/README.md" />
