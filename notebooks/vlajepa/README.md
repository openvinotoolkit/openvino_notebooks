# Robot manipulation with VLA-JEPA and OpenVINO

[VLA-JEPA](https://github.com/ginwind/VLA-JEPA) is a Vision-Language-Action (VLA) policy for
robot manipulation. Given camera views and a natural-language instruction such as *"pick up the black
bowl and place it on the plate"*, it predicts the motion needed to carry out the task —
continuous robot actions rather than text.

The policy has two components:

* **A Qwen3-VL-2B vision-language backbone**, used purely as an encoder. Camera views and the
  instruction are packed into one prompt containing 32 `<|embodied_action|>` placeholder positions,
  whose final hidden states form a compact, action-oriented summary of the scene. Nothing is ever
  decoded to text.
* **A DiT flow-matching action head** (150M parameters) that conditions on those 32 tokens plus the
  robot's own internal state readings and denoises a noise sample into a 7-step action chunk over 4 Euler
  steps. Each step is a 7-DoF vector: 3 translation, 3 rotation, 1 binary gripper command.

```
   camera views ──┐
                  ├──► Qwen3-VL-2B ──► 32 embodied ──► DiT action head ──► 7 × 7-DoF
   instruction ───┘      (encoder)     action tokens    (4 flow-matching     action chunk
                                                          Euler steps)
                              robot state ───────────────────┘
```

## Notebook contents

The tutorial consists of the following steps:

- Install requirements
- Convert both components to OpenVINO IR
- Run inference on a sample observation
- Launch an interactive demo

In this demonstration you provide two camera views and an instruction, and the policy returns a chunk
of future robot actions.

## Requirements

- OpenVINO 2025.0 or newer
- Optimum Intel
- `transformers==4.57.*` (pinned — the checkpoint was validated against this version, and
  `optimum-intel` will otherwise pull `transformers` 5.x)
- Pre-trained VLA-JEPA weights under `pretrained/` (downloaded automatically by the notebook via
  `download_checkpoints.py` if not already present), and the fixed sample observation in
  `golden_outputs/`

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/vlajepa/README.md" />
