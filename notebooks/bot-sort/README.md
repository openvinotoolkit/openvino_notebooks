# BoT-SORT multi-object tracking with OpenVINO

[BoT-SORT](https://github.com/NirAharon/BoT-SORT) uses a YOLOX-X person
detector (`bytetrack_x_mot17.pth.tar`, `1×3×800×1440`) and a FastReID SBS-S50
appearance model (`mot17_sbs_S50.pth`, batches of `8×3×384×128`).
The tracker (camera-motion compensation, Kalman filtering, IoU and appearance
association) runs on the host. Tracking thresholds and MOT17 per-sequence
overrides are kept in the configuration, separate from network weights and IRs.

## Model architecture

![Animated BoT-SORT detector, ReID and tracker pipeline](./botsort_pipeline_animated.svg)

[Open the animated architecture diagram](./botsort_pipeline_animated.svg)
to follow the detector, appearance and motion branches. This original project
drawing also labels INT8 experiments; this tutorial exports **FP32, FP16
and INT8**.

| Stage | Execution | Role |
|---|---|---|
| Frame preparation and YOLOX-X | Host preprocessing, OpenVINO network | Letterbox each frame to `800×1440`; decode person boxes in the detector graph and filter/NMS detections on the host. The detector uses the same MOT17 checkpoint as ByteTrack. |
| FastReID SBS-S50 | Host crops, OpenVINO network | Crop detected people into batches of eight `384×128` RGB images. The network produces 2048-dimensional appearance features, which are normalized for matching. |
| Camera-motion compensation | Host | Estimate camera motion (precomputed GMC for MOT17, ORB for the video) and correct Kalman track predictions. |
| BoT-SORT association | Host | Combine predicted motion, IoU and ReID similarity to associate high-score detections; a second association considers lower-score boxes. Maintain track IDs, lost tracks and new tracks. |

**Model assets:** `bytetrack_x_mot17.pth.tar` holds the YOLOX-X detector
weights; `mot17_sbs_S50.pth` holds the FastReID appearance-network weights.
The notebook exports **both** networks as FP32, FP16 and INT8 `.xml`/`.bin` IRs.
GMC, matching logic and thresholds are not learned network weights: the
tracking configuration is saved alongside and embedded in the IRs.

## Tutorial

The [tutorial](bot-sort.ipynb) loads both original PyTorch
checkpoints for conversion to OpenVINO FP32 and FP16 IR, calibrates both
networks to INT8 using 32 spaced frames and detected-person crops from the
bundled video, verifies inference, can evaluate MOT17 val-half MOTA/IDF1
and tracks a video at each precision.
[botsort_nb_utils.py](botsort_nb_utils.py) contains the host-side tracker,
ReID crops, preprocessing and evaluation. No separate PyTorch notebook or
benchmark script is needed in this folder.

## Installation Instructions and data

Run the tutorial **from this folder**. Its first code cell installs Python
dependencies using `%pip`. On first use, the notebook clones the public
BoT-SORT source and downloads its official detector and ReID checkpoints
into the ignored `model/` directory. IRs and results go to the ignored
`output/` directory.

The [bundled video](pedestrians_cc0_30s.mp4) contains exactly **30 seconds
(900 frames at 30 FPS)**. All three OpenVINO variants track at most those 900
frames at **FP32, FP16 and INT8**; a playable annotated MP4 and a contact sheet
are saved under `output/results/video/` for each precision. The contact sheets
and embedded playable videos remain visible in the executed notebook. GPU FP32 explicitly
requests `f32` execution. Video calibration is a self-contained INT8
demonstration, not a substitute for independent MOT17 accuracy measurement.
To change the inference device, edit the dropdown's `value` in the notebook's
device-selection cell. Saved cell outputs use plain text rather than widget views,
so results remain readable in viewers without Jupyter widget support.

MOT17 is not redistributed here. To run the optional accuracy section,
provide the MOT17 FRCNN train sequences in `data/MOT17/train/` and set
`RUN_MOT17_ACCURACY = True` in the notebook.

**Video license:** The bundled excerpt is derived from
[DiagonalCrosswalkYongeDundas.webm](https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm)
by Raysonho @ Open Grid Scheduler / Grid Engine (own work, [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)).
It was resized, encoded to H.264 and limited to 900 frames at 30 FPS, with the
audio removed. This public-domain pedestrian scene is used instead of the
unlicensed local demonstration video.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/bot-sort/README.md" />
