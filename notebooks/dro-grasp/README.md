# DRO-Grasp with OpenVINO

This tutorial converts the [DRO-Grasp](https://github.com/mhpanah/DRO-Grasp) PyTorch validation network to OpenVINO IR and runs deterministic inference on Intel CPU and GPU.

The notebook covers:

- downloading the pinned DRO-Grasp source and released checkpoint;
- exposing an explicit latent input and tensor-only output for deterministic conversion;
- converting with `ov.convert_model` and saving explicit FP32 and FP16 OpenVINO IR variants;
- replacing mutation-based FP16 block assembly with a scatter-free concatenation graph;
- validating both dynamic models with 32-point and published 512-point inputs;
- comparing PyTorch with OpenVINO FP32 and FP16 numerical accuracy;
- benchmarking PyTorch CPU and OpenVINO FP32/original FP16/optimized FP16 CPU/GPU latency; and
- generating a host-specific implementation report.

## Run

Follow the [OpenVINO Notebooks installation guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide), open [dro-grasp.ipynb](./dro-grasp.ipynb), and execute it from top to bottom. The prerequisites cell installs the notebook dependencies.

The released model and checkpoint are downloaded automatically. Generated source, checkpoint, FP32/FP16 IR, and host-specific report files are excluded from version control.

## Validated results
Accuracy in this tutorial means numerical agreement with PyTorch. Full grasp-success evaluation additionally requires DRO-Grasp datasets, robot assets, the optimization pipeline, and CUDA-only Isaac Gym.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/dro-grasp/README.md" />
