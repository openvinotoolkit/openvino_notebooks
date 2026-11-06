# Bimanual robot control with LingBot-VLA v2 and OpenVINO

[LingBot-VLA v2](https://github.com/Robbyant/lingbot-vla-v2) is a Vision-Language-Action (VLA) policy from Robbyant.
Given images from the robot cameras, the robot joint state and a task instruction, it predicts a chunk of the next
50 robot actions. The notebook uses the 6B checkpoint fine-tuned on the RoboTwin 2.0 dual-arm benchmark, where each
action holds 6 joint positions and 1 gripper position per arm.

The policy combines a Qwen3-VL-4B vision-language backbone, which encodes the three camera views and the instruction
once, with a mixture-of-experts flow-matching action expert, which turns Gaussian noise into an action chunk in
10 Euler steps while attending to the backbone's key/value cache.

## Notebook contents

The tutorial consists of the following steps:

- Install requirements and download the model
- Load one recorded episode of the [RoboTwin 2.0 dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) and preprocess its first frame
- Generate PyTorch reference actions and convert the whole `sample_actions` call to a single OpenVINO model
- Compress the weights with NNCF (INT4 for the MoE experts, INT8 for the other layers)
- Run inference with OpenVINO, compare the actions with the PyTorch reference and measure latency
- Replay the episode through the OpenVINO model and render the predicted and recorded joint trajectories as a video

## Requirements

- Linux (the upstream code imports Triton)
- At least 64 GB of RAM and about 50 GB of free disk space
- An Intel GPU is recommended: the FP16 model needs about 13 GB of GPU memory, the compressed one about 6 GB

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/lingbot-vla-v2/README.md" />
