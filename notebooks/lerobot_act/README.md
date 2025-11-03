# ACT Policy → OpenVINO IR Conversion (Notebook Guide)

This README documents the current workflow implemented in `act_to_openvino.ipynb` for converting a LeRobot ACT (Action Chunking Transformer) PyTorch checkpoint into an OpenVINO IR (XML/BIN) model. The notebook presently performs FP32 export (Model Optimizer invoked with FP16 compression flag but output standardized to `act_model_fp32.xml/bin`).


## Required Checkpoint Files (`act_checkpoint/`)
Place these next to the notebook:
* `model.safetensors` – ACT weights
* `config.json` – architecture + feature definitions
* `train_config.json` – optional (reproducibility record)
* `stats.json` – optional normalization statistics

## Minimal Installation & Launch
Recommended (Conda environment):
```bash
conda create -n unitree_lerobot python=3.10 -y
conda activate unitree_lerobot
# Optional helper script if present in this folder
bash setup_unitree_lerobot_env.sh
# Launch notebook with correct kernel
jupyter lab act_to_openvino.ipynb --NotebookApp.kernel_name=unitree_lerobot
```

If you skip creating a dedicated environment, the dependency cell will install core packages (torch, openvino, nncf, etc.) into the current kernel. You MUST still install `lerobot` manually; the notebook will not auto‑install it.


## Key Configuration Variables
| Variable          | Meaning                                                  |
|-------------------|----------------------------------------------------------|
| `CKPT_DIR`        | Relative checkpoint folder (`act_checkpoint`)            |
| `CHECKPOINT_PATH` | Path to `model.safetensors` (env‑overrideable)           |
| `IR_OUTPUT_DIR`   | Destination for `model.onnx` & IR artifacts              |
| `STATS_PATH`      | Path to `stats.json` if present                          |
| `PRECISIONS`      | Currently `['FP32']`                                     |
| `TARGET_DEVICE`   | Default runtime device                                   |

## ONNX Export
Wrapper (`ONNXWrapper`) mirrors ACT forward usage by constructing a batch dict. Input ordering:
`observation.state`, each camera image (`observation.images.*`), `action_is_pad`, `action`, optional `observation.environment_state`.
* Output name: `output`
Exports only if `openvino_ir_outputs/model.onnx` does not already exist.

## Model Optimizer Conversion
Executed command:
```
mo --input_model openvino_ir_outputs/model.onnx --output_dir openvino_ir_outputs --compress_to_fp16=False
```
Artifacts are copied / renamed to:
* `act_model_fp32.xml`
* `act_model_fp32.bin`

## Directory Layout After Successful Conversion
```
act_to_openvino.ipynb
act_checkpoint/
  model.safetensors
  config.json
  train_config.json
  stats.json (optional)
openvino_ir_outputs/
  model.onnx
  act_model_fp32.xml
  act_model_fp32.bin
```