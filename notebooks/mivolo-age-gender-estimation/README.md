# MiVOLO v2 conversion and validation

This notebook converts MiVOLO v2 and the YOLOv8 person/face detector to OpenVINO
FP16-weight IR, quantizes MiVOLO to mixed INT8/FP16 with NNCF, and checks numerical
conversion and held-out accuracy. YOLO is converted to FP16 only.
An interactive image demo and FP16/INT8 inference benchmarks are also included.

## Table of contents

- [About MiVOLO v2](#about-mivolo-v2)
- [Model references](#model-references)
- [Example result](#example-result)
- [Installation instructions](#installation-instructions)
- [Outputs](#outputs)
- [Visualization and benchmarking](#visualization-and-benchmarking)
- [Accuracy and validation](#accuracy-and-validation)
- [PR boundary](#pr-boundary)

## About MiVOLO v2

[MiVOLO v2](https://huggingface.co/iitolstykh/mivolo_v2) is a multi-input
vision transformer for age estimation and gender classification. It combines
information from a person's face and body crops through cross-attention,
using appearance cues from both regions rather than relying on the face alone.
The model predicts a continuous age value and gender-class scores; these are
model estimates, not verified personal attributes.

MiVOLO v2 consumes 384 x 384 crops. In this notebook, a YOLOv8 person/face
detector first locates subjects, and the associated face and body crops are
normalized and concatenated into a six-channel model input. OpenVINO runs the
converted estimator, while NNCF provides accuracy-controlled INT8 quantization
for comparison with the FP16-weight baseline.

For the architecture and research background, see the
[MiVOLO source repository](https://github.com/WildChlamydia/MiVOLO),
[original MiVOLO paper](https://arxiv.org/abs/2307.04616), and
[MiVOLOv2 paper](https://arxiv.org/abs/2403.02302).

## Model references

| Model | Reference | Revision used by this notebook |
| --- | --- | --- |
| MiVOLO v2 age/gender estimator | [Hugging Face model card](https://huggingface.co/iitolstykh/mivolo_v2), [source code](https://github.com/WildChlamydia/MiVOLO) | [53393526c220e34cdd7b722b36d22b6f9e5f4241](https://huggingface.co/iitolstykh/mivolo_v2/tree/53393526c220e34cdd7b722b36d22b6f9e5f4241) |
| YOLOv8x person/face detector | [Hugging Face model card](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector), [Ultralytics source code](https://github.com/ultralytics/ultralytics/tree/v8.1.0) | [b3d071aedccd46a3b2d4b40609da6880a815f395](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector/tree/b3d071aedccd46a3b2d4b40609da6880a815f395) |

Research references:

- Maksim Kuprashevich and Irina Tolstykh,
  [MiVOLO: Multi-input Transformer for Age and Gender Estimation](https://arxiv.org/abs/2307.04616).
- Maksim Kuprashevich and Irina Tolstykh,
  [MiVOLOv2: Revisiting Age and Gender Estimation with Improved Generalization and Embeddings](https://arxiv.org/abs/2403.02302).

## Example result

![MiVOLO demo showing detected face/body boxes and predicted age and gender](screenshot.png)

Representative output from the OpenVINO INT8 demo on CPU. Labels are model
predictions, not verified attributes; results may vary with device and precision.
Sample photograph: Pete Souza / White House, via
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:President_Barack_Obama.jpg).
Step 5 saves the displayed figure as `screenshot.png` in the notebook working
directory, so you can regenerate this example after running the demo.

## Installation instructions

This is a self-contained example. We recommend running it in a dedicated virtual
environment with Jupyter available.

For general environment setup, see the main
[OpenVINO Notebooks Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md).

Open [mivolo-v2.ipynb](mivolo-v2.ipynb) and run all cells in order.

Model code/checkpoints and dataset are pinned to immutable Hugging Face revisions.
MiVOLO loads reviewed remote code and the detector checkpoint uses pickle: only
use trusted revisions. The first run needs network access and downloads large
weights/data. `FORCE_MIVOLO_EXPORT=1` rebuilds the MiVOLO export.
Conversion metadata prevents reuse of previously optimized MiVOLO FP16/INT8 IRs.

## Outputs

Outputs under `mivolo_workspace/` are:

| Artifact | Purpose |
| --- | --- |
| `models_fp16/mivolo_v2.xml` + `.bin` | MiVOLO FP16 weights, normalized float NCHW input `[1,6,384,384]` |
| `models_int8/mivolo_v2.xml` + `.bin` | Accuracy-controlled mixed-precision MiVOLO, same I/O |
| `models_fp16/yolov8x_person_face.xml` + `.bin` | Static YOLO FP16 detector `[1,3,640,640]` |

No `_deploy` models or custom optimization implementations are generated.
Weight precision is not an execution precision guarantee.
The source model forward methods are unchanged; no custom layout rewrites,
weight folding, MVN replacement, attention quantization exclusions, or performance
configuration is applied to conversion or the demo. The benchmark uses standard
LATENCY/THROUGHPUT hints solely for measurement.
Standard exporter/OpenVINO transformations and NNCF
accuracy-controlled quantization remain part of the conversion workflow.

## Visualization and benchmarking

After conversion/validation, select an available device (CPU, GPU, NPU or AUTO)
and FP16/INT8 estimator precision, then execute the demo cells. It downloads a
sample portrait; replace `image_path` with your own image to use another input.
Rerun the inference cell after changing a widget. Detections, face/body association,
host-side crop normalization and age/gender decoding use the raw converted models.
The annotated image shows face/body boxes with readable gender, confidence and
age labels overlaid on each detected face. Device/operator
compilation errors are surfaced rather than hidden.

The final cell invokes OpenVINO's benchmark utility for each MiVOLO precision
with latency and throughput hints, for 10 seconds per configuration. AUTO is
passed through to OpenVINO. Results exclude host preprocessing and the detector;
they are not full-pipeline latency or an accuracy measurement.

## Accuracy and validation

- NNCF calibration: first 300 usable FairFace training samples.
- Accuracy control: next 150 training samples (disjoint from calibration).
- Final evaluation: 250 samples from the separate validation split.
- MiVOLO original PyTorch vs exported IR: three validation samples, raw outputs,
  `atol=0.02`, `rtol=0.02` (device floating-point tolerance).
- YOLO: deterministic synthetic-input raw-output smoke check, `atol=0.5`,
  `rtol=0.03`; this is not labeled detection accuracy.
- Final INT8 checks fail above 2 percentage points of gender-accuracy loss or
  0.75 years of additional age MAE. NNCF controls age MAE only; the final check
  independently enforces both budgets. Failure is not silently accepted.

FairFace age labels are buckets; midpoint MAE is only a proxy for exact-age MAE.
The notebook uses fixed face regions and whole-image body crops, not a labeled
end-to-end detection dataset. The small evaluation is a regression check, not
a production demographic-fairness or generalization certification.

## PR boundary

**Notebook PR:** this directory (generator, notebook, README and tests). It has no
dependency on the optimization preparation deliverable.
**Optimization preparation PR:** the separate `toolkit_optimizations/` directory
retains the exporter/IR rewrites, tests and performance benchmark. It is not used
by this notebook and is not a native OpenVINO toolkit pass/kernel implementation.

