# MedASR Medical Speech Recognition with OpenVINO

This notebook demonstrates converting Google's MedASR (Medical Automatic Speech Recognition) model to OpenVINO format with FP16 and INT8 quantization for efficient medical speech-to-text transcription.

## Overview

MedASR is a specialized speech recognition model optimized for medical terminology. This tutorial shows how to:

- Load the MedASR model from HuggingFace
- Convert it to OpenVINO IR format for optimal inference performance
- Apply INT8 quantization using NNCF for model compression
- Compare accuracy and performance across PyTorch, FP16, and INT8 versions

## Key Features

- **Model Compression**: 3.9x size reduction (402 MB → 102 MB) with INT8 quantization
- **High Accuracy**: 97.98% token-level accuracy maintained after INT8 quantization
- **Medical Terminology**: Optimized for accurate medical speech recognition

## Tutorial Contents

1. **Installation** - Install required packages (OpenVINO, NNCF, Transformers, etc.)
2. **Load Model** - Load Google's MedASR model from HuggingFace
3. **Prepare Audio Data** - Download and preprocess test audio (optimized for 10s chunks)
4. **PyTorch Inference** - Establish baseline accuracy with original model
5. **Convert to OpenVINO FP16** - Convert using torch.export and ov.convert_model
6. **INT8 Quantization** - Apply NNCF quantization with real audio calibration
7. **Accuracy Comparison** - Validate quantization quality across all versions
8. **Performance Benchmarking** - Measure inference speed on CPU and GPU

## Results

- **Model Size**: 402 MB (FP16) → 102 MB (INT8) = **3.9x compression**
- **Accuracy**: 97.98% token match between INT8 and PyTorch
- **Model Shape**: Static [1, 998, 128] optimized for 10-second audio chunks

## Installation

```bash
pip install -q "openvino>=2024.4.0" "nncf>=2.13.0" "torch>=2.1" "transformers>=5.4.0" "librosa" "soundfile" "huggingface_hub"
```

## Important Notes

⚠️ **Gated Model Access**: The MedASR model is gated on HuggingFace. You must:
1. Request access at https://huggingface.co/google/medasr
2. Authenticate with your HuggingFace token before running the notebook

## Use Cases

- Medical transcription systems
- Clinical documentation automation
- Healthcare voice assistants
- Medical education and training platforms
