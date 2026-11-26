# Vision-language navigation with InternVLA-N1 DualVLN and OpenVINO

[InternVLA-N1](https://github.com/InternRobotics/InternNav) is a vision-language navigation (VLN)
foundation model. Given a natural-language route description and the robot's camera stream, it decides
how the robot should move next. The [DualVLN](https://arxiv.org/abs/2512.08186) variant splits that
decision between two systems:

* **System 2, a Qwen2.5-VL-7B planner**, reads the instruction, past frames and the current frame, and
  generates either discrete moves (forward, turn left/right, look down, stop) or a pixel goal in the image.
  For a pixel goal, a second prefill with 4 learned query tokens produces the trajectory latents.
* **System 1, a lightweight trajectory generator**, encodes two RGB frames with DINOv2-S and a Q-Former,
  conditions a NextDiT on them and the trajectory latents, and denoises 32 candidate trajectories over 10
  flow-matching steps. Their mean is converted into local moves.

```
                                       ┌──► discrete moves (↑ ← → ↓ STOP)
  instruction ──┐                      │
                ├──► Qwen2.5-VL-7B ────┤
  RGB frames ───┘      (System 2)      └──► pixel goal ──► 4 trajectory latents ──┐
                                                                                  ▼
                 2 RGB frames ──► DINOv2 + Q-Former ──► 36 context ──► NextDiT ──► 32 candidate ──► local
                                    (System 1)           tokens      (10 Euler   trajectories     moves
                                                                      steps)
```

## Notebook contents

The tutorial consists of the following steps:

- Install requirements
- Generate a PyTorch reference on a fixed sample observation
- Convert both systems to OpenVINO IR
- Run System 2 planning and System 1 trajectory generation with OpenVINO
- Validate each stage against the PyTorch reference

In this demonstration you provide camera frames and an instruction; System 2 returns a plan, and System 1
returns candidate trajectories and the local moves that follow them.

## Requirements

- OpenVINO 2025.4 or newer
- Optimum Intel
- `transformers==4.57.*` (pinned — the checkpoint loader and export were validated against this version,
  and `optimum-intel` will otherwise pull `transformers` 5.x)
- The InternVLA-N1-DualVLN checkpoint under `pretrained/` (downloaded automatically by the notebook via
  `download_checkpoints.py` if not already present)

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/internvla-n1-dualvln/README.md" />
