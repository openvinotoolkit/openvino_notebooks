# ACT Policy → OpenVINO IR Conversion (Notebook Guide)
This README documents the current workflow implemented in `lerobot-act.ipynb` for converting a LeRobot ACT (Action Chunking Transformer) PyTorch checkpoint into an OpenVINO IR (XML/BIN) model.

# Run
`jupyter lab lerobot-act.ipynb`

## Required Checkpoint Files (`act_checkpoint/`)
Place these next to the notebook:
* `model.safetensors` – ACT weights
* `config.json` – architecture + feature definitions
* `train_config.json` – optional (reproducibility record)
* `stats.json` – optional normalization statistics

## Required dataset Files (`dataset/G1_BlockStacking_Dataset/`)
Download the G1_BlockStacking_Dataset from hugging face:
https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset


## Key Configuration Variables
| Variable          | Meaning                                                  |
|-------------------|----------------------------------------------------------|
| `CKPT_DIR`        | Relative checkpoint folder (`act_checkpoint`)            |
| `CHECKPOINT_PATH` | Path to `model.safetensors` (env‑overrideable)           |
| `IR_OUTPUT_DIR`   | Destination for `model.onnx` & IR artifacts              |
| `STATS_PATH`      | Path to `stats.json` if present                          |
| `PRECISIONS`      | Currently `['FP32']`                                     |
| `TARGET_DEVICE`   | Default runtime device                                   |


## Direct PyTorch FX Conversion
Instead of exporting full temporal tensors via ONNX you can generate a smaller IR directly from PyTorch using OpenVINO's FX path. The wrapper internally creates placeholder temporal inputs (`action`, `action_is_pad`, history) so the IR exposes only observation features:
* `observation_state`
* `observation_images_0..N` (one input per camera)

Resulting files:
* `act_model_direct_fp32.xml/bin`
* `act_model_direct_fp16.xml/bin`

## INT8 Quantization (NNCF)
You can produce an INT8 version for reduced size / latency using NNCF post‑training quantization.

Prerequisites:
* Direct FP32 IR: `act_model_direct_fp32.xml`
* Representative dataset root (`ACT_DATASET_ROOT`) with episodes
* Normalization stats: `stats.json`

Generated files:
* `openvino_ir_outputs/int8/model_int8.xml/bin`

Tips:
* Increase calibration samples for better accuracy.
* Use `preset='accuracy'` if performance preset degrades results too much.
* Ensure OpenVINO and NNCF versions are compatible (>= 2025.0.0 for OpenVINO runtime if using latest NNCF).


## Evaluation of Variants
The notebook / helper script can compare PyTorch baseline vs IR variants (Direct FP32, FP16, INT8).

Environment variables (set before running evaluation cell):
| Var | Purpose |
|-----|---------|
| `OPENVINO_MODEL_PATH` | Path to IR `.xml` file to evaluate |
| `STATS_PATH` | Path to `stats.json` for normalization |
| `OPENVINO_DEVICE` | `CPU|GPU|NPU|AUTO` (compile target) |
| `OPENVINO_PRECISION_HINT` | Optional override (`FP32|FP16|INT8`) |


Evaluation pipeline steps:
1. Load PyTorch ACT and normalization stats.
2. Compile OpenVINO model.
3. Run action predictions over dataset episodes.
4. Apply optional temporal smoothing ensemble.
5. Plot per‑joint trajectories & error statistics (saved as `actions_comparison_<variant>.png`).


## Directory Layout (Example After Conversion, FP16 & INT8 Quantization)
```
lerobot-act.ipynb
act_checkpoint/
  model.safetensors
  config.json
  train_config.json
  stats.json              # normalization (recommended, required for eval & INT8)
dataset/
  G1_BlockStacking_Dataset/
openvino_ir_outputs/
  act_model_direct_fp32.xml  # Direct minimal-input IR
  act_model_direct_fp32.bin
  int8/
    model_int8.xml          # Post-training quantized INT8 IR
    model_int8.bin

actions_comparison_direct_fp32.png
actions_comparison_direct_fp16.png
actions_comparison_int8.png

```