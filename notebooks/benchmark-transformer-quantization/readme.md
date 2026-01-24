# ⚡ Transformer Quantization Benchmark: PyTorch vs. OpenVINO

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2025.0-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

A modular benchmarking framework designed to measure the **inference performance gap** between standard PyTorch (FP32) and optimized OpenVINO (INT4) runtimes for Large Language Models (LLMs) on CPU.

> **Key Result:** Achieved a **~10.39x Speedup** and **3.1x Storage Reduction** on Qwen2.5-0.5B-Instruct.

## 🚀 Overview

Running LLMs on consumer hardware (laptops, edge devices) is challenging due to high latency and memory usage. This project demonstrates how **NNCF (Neural Network Compression Framework)** and **INT4 Quantization** can unlock real-time performance on standard CPUs.

**This benchmark measures:**
* **Latency (P50):** Median time to generate a token (Chat responsiveness).
* **Throughput:** Total tokens generated per second.
* **Memory Footprint:** RAM usage during generation.
* **Disk Size:** Exact storage efficiency (Compression Rate) calculated via physics-based analysis.

## ✨ Key Features
* **Self-Contained Setup:** No manual terminal commands required. The notebook automatically detects and installs all necessary dependencies (`openvino`, `torch`, `optimum-intel`).
* **Interactive Controls:** Built-in widgets allow you to select your target **Device** (CPU/GPU) and **Quantization Precision** (INT4, INT8, FP16) dynamically.
* **Robust Metrics:** Implements aggressive Garbage Collection (GC) and "Physics-Based" size calculation to ensure 100% accurate comparisons across different hardware.

## 📊 Benchmark Results (Sample)

![alt text](image.png)

## 📂 Project Structure

```text
benchmark-transformer-quantization/
├── bench/
│   ├── inputs.py         # Input processing & tokenization helpers
│   ├── kv_cache.py       # Memory management & Garbage collection
│   ├── metrics.py        # System-level monitoring (RAM/Disk)
│   ├── model_loader.py   # Handles FP32 loading & INT4 export
│   ├── quantization.py   # NNCF quantization logic & compression
│   ├── runner.py         # Warmup & Measurement loop
│   └── utils.py          # General utility functions
├── configs/
│   └── benchmark_config.yaml  # Easy-to-tune parameters
├── benchmark-transformer-notebook.ipynb  # 📖 Main Interactive Notebook
├── image.png             # Result visualization
└── README.md             # overview of whole folder