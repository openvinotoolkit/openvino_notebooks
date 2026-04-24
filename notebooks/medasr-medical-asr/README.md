# MedASR Medical Speech Recognition with OpenVINO

[MedASR](https://huggingface.co/google/medasr) is a specialized Automatic Speech Recognition (ASR) model from Google, optimized for medical terminology. It is a CTC-based model built on top of the Conformer architecture.

This notebook demonstrates converting Google's MedASR model to OpenVINO format with FP16 and INT8 quantization for efficient medical speech-to-text transcription.

## Notebook Contents

The tutorial consists of the following steps:

1. **Installation** - Install required packages (OpenVINO, NNCF, Transformers, etc.)
2. **Login to HuggingFace** - Authenticate to access the gated model
3. **Load Model** - Load Google's MedASR model from HuggingFace
4. **Prepare Audio Data** - Download and preprocess test audio (optimized for 10s chunks)
5. **PyTorch Inference** - Establish baseline accuracy with original model
6. **Convert to OpenVINO FP16** - Convert using torch.export and ov.convert_model
7. **INT8 Quantization** - Apply NNCF quantization with real audio calibration
8. **Accuracy Comparison** - Validate quantization quality across all versions
9. **Performance Benchmarking** - Measure inference speed on the selected device

## Important Notes

⚠️ **Gated Model Access**: The MedASR model is gated on HuggingFace. You must:
1. Request access at https://huggingface.co/google/medasr
2. Authenticate with your HuggingFace token before running the notebook

## Installation Instructions

This is a self-contained example that relies solely on its code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/medasr-medical-asr/README.md" />
