# What this notebook does

This notebook provides a high-level overview of working with Intel GPUs in OpenVINO. It shows how to use Query Device to list system GPUs and check their properties, and it explains some of the key properties. It shows how to compile a model on GPU with performance hints and how to use multiple GPUs using MULTI or CUMULATIVE_THROUGHPUT.

The notebook also presents example commands for benchmark_app that can be run to compare GPU performance in different configurations. It also provides the code for a basic end-to-end application that compiles a model on GPU and uses it to run inference.

This notebook provides a practical, engineering‑focused introduction to using **Intel® GPU devices** with OpenVINO. It walks through device discovery, GPU property inspection, idempotent model download, model loading, and GPU‑targeted compilation. It also demonstrates how to run inference on a sample image or video and how to fall back to CPU automatically if no GPU is available.

# Hardware & device support

This notebook supports the following devices:

- CPU — **supported (fallback)**
- GPU — **primary target**
- NPU — **supported if available**

# Device fallback logic

```python
import openvino as ov

core = ov.Core()
available = core.available_devices
device = (
    "NPU" if "NPU" in available
    else "GPU" if "GPU" in available
    else "CPU"
)
print(f"Selected device: {device}")
```

# Setup

Make sure that **uv** is installed.

## Windows
irm https://astral.sh/uv/install.ps1 | iex

## macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

Verify installation:
uv --version

## Install dependencies

From the gpu-device notebook:

uv sync

uv run jupyter lab gpu-device.ipynb

# Expected output

When the notebook runs successfully, you should see:

- **Device selection output**, e.g.  
  `Selected device: GPU` or `Selected device: CPU` depending on availability.

- **Model download logs**, e.g.  
  `Model not found — downloading from Hugging Face Hub...`  
  followed by Hugging Face progress bars.

- **Model loading and compilation**, e.g.  
  `Compiled model on GPU`

Exact values may vary depending on hardware and OpenVINO version.

# Tested‑on

| OS | Python | OpenVINO | Device(s) | Status |
|----|--------|----------|-----------|--------|
| Windows 11 | 3.12 | 2026.2 | CPU, GPU | Pass |

# Troubleshooting

### GPU not detected  
**Cause:** Missing or outdated GPU driver.  
**Fix:** Install the latest Intel® Graphics driver:  
https://www.intel.com/content/www/us/en/download-center/home.html

### Model download warnings  
**Cause:** Hugging Face deprecation notices.  
**Fix:** Safe to ignore — does not affect execution.

### CPU selected even though GPU is present  
**Cause:** Permissions, driver issues, or unsupported hardware.  
**Fix:**  
- Update GPU drivers  
- Verify `ov.Core().available_devices`  
- Ensure the notebook is running inside the correct environment

### OpenCV or video display issues  
**Cause:** Missing codecs or unsupported environment.  
**Fix:**  
- Try a static image instead of video  
- Ensure OpenCV is installed correctly

# References

- Upstream OpenVINO notebooks:  
  https://github.com/openvinotoolkit/openvino_notebooks

- OpenVINO GPU documentation:  
  https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html

- OpenVINO performance hints:  
  https://docs.openvino.ai/2024/openvino-workflow/running-inference/performance-hints.html
