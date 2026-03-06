# Imitation Learning with ACT (Action Chunking Transformer)

This notebook demonstrates how to use the ACT (Action Chunking Transformer) model for imitation learning tasks for Aloha Robot, with OpenVINO optimization for efficient inference.

**Note:** This notebook currently supports **Ubuntu 22.04**, Python 3.10-3.12 only.

## Overview

Imitation learning is a machine learning approach where a model learns to mimic expert behavior by observing and replicating demonstrations. ACT is an action chunking policy that uses Transformers for sequence modeling and is trained as a conditional VAE (CVAE) to capture variability in human data. It significantly outperforms previous imitation learning algorithms on various simulated and real-world fine manipulation tasks.

For more details, see the original paper: [Action Chunking Transformer](https://arxiv.org/pdf/2304.13705)

## Contents

This notebook covers:

1. **Prerequisites** - Downloading pre-trained weights
2. **Dependency Installation** - Setting up required packages and repositories
3. **Model Conversion** - Converting PyTorch checkpoint to OpenVINO IR format
4. **Device Selection** - Choosing target hardware for inference
5. **Policy Evaluation** - Running the trained policy on simulation tasks

## Requirements

- OpenVINO 2025.4.1
- PyTorch 2.7.1
- MuJoCo 3.2.6
- Additional dependencies (automatically installed by the notebook)

## Pre-trained Weights

Pre-trained weights for the `sim_insertion_scripted` task are required. The notebook will automatically download and extract them from:
- [Download Link](https://eci.intel.com/embodied-sdk-docs/_downloads/sim_insertion_scripted.zip)

## Installation

The notebook handles all installation automatically, including:

- Cloning the edge-ai-suites repository (sparse checkout)
- Setting up the ACT pipeline from the robotics-ai-suite
- Installing DETR (Detection Transformer) dependencies
- Applying OpenVINO-specific patches
- Installing all required Python packages

## Usage

Simply run the notebook cells in order:

1. **Download Pre-trained Weights** - Downloads and extracts model checkpoint
2. **Install Dependencies** - Sets up environment and installs packages
3. **Convert Model** - Converts PyTorch checkpoint to OpenVINO IR format
4. **Select Device** - Choose CPU, GPU, or other available hardware
5. **Evaluate Policy** - Run inference and evaluate the policy performance

## Model Conversion

The conversion uses the following parameters:
```bash
python3 ov_convert.py --ckpt_path <path> --height 480 --weight 640 --camera_num 4 --chunk_size 100
```

## Evaluation

The evaluation script runs with these default parameters:
```bash
python3 imitate_episodes.py \
  --task_name sim_insertion_scripted \
  --ckpt_dir <path> \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0 \
  --device CPU/GPU \
  --eval
```

## Repository Structure

```
aloha-act/
├── aloha-act.ipynb           # Main notebook
├── README.md                  # This file
├── sim_insertion_scripted/    # Downloaded pre-trained weights
│   └── four_camera/
│       └── policy_best.ckpt
└── edge-ai-suites/            # Cloned ACT pipeline
    └── robotics-ai-suite/
        └── pipelines/
            └── act-sample/
```

## References

- [ACT Paper](https://arxiv.org/pdf/2304.13705) - Action Chunking Transformer
- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [Edge AI Suites](https://github.com/open-edge-platform/edge-ai-suites)
- [Original ACT Implementation](https://github.com/tonyzhaozh/act)

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/aloha-act/README.md" />
