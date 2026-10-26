# ByteTrack multi-object tracking with OpenVINO

[ByteTrack](https://github.com/FoundationVision/ByteTrack) combines a YOLOX-X
person detector (MOT17 checkpoint `bytetrack_x_mot17.pth.tar`, input
`1×3×800×1440`) with a host-side BYTETracker. Detection decoding is included
in the exported network; filtering, association, Kalman filtering and
per-sequence tracking thresholds are **not** in the weights or the IR. The
tracker settings accompany the IR as `bytetrack_config.json` and model metadata.

## Model architecture

![Animated ByteTrack detector and tracker pipeline](./bytetrack_pipeline_animated.svg)

[Open the animated architecture diagram](./bytetrack_pipeline_animated.svg)
to see the per-frame flow and the track-state feedback loop. The drawing was
made for the broader ByteTrack study, so the asset paths illustrated there
are examples rather than additional tutorial requirements.

| Stage | Execution | Role |
|---|---|---|
| Image preparation | Host | Resize and letterbox a video/MOT17 frame to `800×1440`, retaining the scale factor needed to map boxes back to the original frame. |
| YOLOX-X | PyTorch for export; OpenVINO for inference | Run the single-class person detector. Anchor-free grid decoding is folded into the network; output boxes, objectness and class scores feed the tracker. |
| Detection filtering | Host | Apply confidence filtering and non-maximum suppression (NMS); restore image coordinates. |
| BYTETracker | Host | Predict existing tracks with a Kalman filter, associate high-confidence detections first and then low-confidence detections by IoU/Hungarian matching. Create, retain or retire track IDs according to the tracker thresholds and buffer. |

**Model assets:** `bytetrack_x_mot17.pth.tar` contains the YOLOX-X detector
weights; it does not contain BYTETracker parameters. The tutorial creates
FP32, then FP16 and NNCF INT8 OpenVINO `.xml`/`.bin` detector IRs. It stores
`bytetrack_config.json` alongside each IR and embeds the same configuration
in the IR metadata. There is **no ReID network** in ByteTrack.

## Tutorial

The [tutorial](byte-track.ipynb) converts the YOLOX-X detector to OpenVINO
FP32 and FP16 IR, quantizes it to INT8 with NNCF, checks accuracy against the
PyTorch reference on MOT17 val-half, benchmarks each precision with
`benchmark_app` and tracks a video at FP32, FP16 and INT8. The executed
notebook keeps its cell outputs.
[bytetrack_nb_utils.py](bytetrack_nb_utils.py) and [bt_common.py](bt_common.py)
contain the host-side pre/postprocessing, tracker and evaluation shared by all
three OpenVINO precisions.

## Installation Instructions and data

Run the tutorial **from this folder**. Install the Python packages the
notebook imports (OpenVINO, NNCF, PyTorch, OpenCV, motmetrics, lap, yacs,
ipywidgets, matplotlib) and provide the ByteTrack source clone, the official
MOT17 detector checkpoint (`bytetrack_x_mot17.pth.tar`) and the MOT17 FRCNN
train sequences. The notebook was written against a project layout that also
holds the PyTorch baseline artifacts (`artifacts/`) produced by its companion
PyTorch notebook, so a fresh clone cannot re-run it end to end without them.
The saved outputs show a complete run.

The bundled [JK_Standing_Next2U.mp4](JK_Standing_Next2U.mp4) is the first
**30 seconds (900 frames at 30 FPS)** of the demonstration video, re-encoded
at 1920x1080. The annotated outputs of the notebook run are in
[results/video/](results/video):
[OV_FP32_tracked.mp4](results/video/OV_FP32_tracked.mp4),
[OV_FP16_tracked.mp4](results/video/OV_FP16_tracked.mp4) and
[OV_INT8_tracked.mp4](results/video/OV_INT8_tracked.mp4)
(900 frames, 960x536).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/byte-track/README.md" />
