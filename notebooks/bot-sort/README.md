# BoT-SORT multi-object tracking with OpenVINO

[BoT-SORT](https://github.com/NirAharon/BoT-SORT) uses a YOLOX-X person
detector (`bytetrack_x_mot17.pth.tar`, `1×3×800×1440`) and a FastReID SBS-S50
appearance model (`mot17_sbs_S50.pth`, batches of `8×3×384×128`).
The tracker (camera-motion compensation, Kalman filtering, IoU and appearance
association) runs on the host. Tracking thresholds and MOT17 per-sequence
overrides are kept in the configuration, separate from network weights and IRs.

## Pipeline

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

The [tutorial](bot-sort.ipynb) is self-contained: it clones the BoT-SORT
repository, downloads the two official checkpoints, builds the PyTorch FP32
networks, converts both to OpenVINO FP32 and FP16 IR, quantizes them to INT8
with NNCF, benchmarks each precision with `benchmark_app`-style runs and
tracks a video at FP32, FP16 and INT8, scoring every precision against
PyTorch FP32 reference tracks computed in the notebook.
[botsort_nb_utils.py](botsort_nb_utils.py) contains the host-side tracker,
ReID crops, preprocessing and evaluation.

## Installation Instructions and data

Run the tutorial **from this folder**. Install the Python packages the
notebook imports (OpenVINO, NNCF, PyTorch, OpenCV, motmetrics, lap, yacs,
ipywidgets, matplotlib, gdown, imageio-ffmpeg). The BoT-SORT source clone and
the two official checkpoints (`bytetrack_x_mot17.pth.tar`,
`mot17_sbs_S50.pth`) are downloaded automatically on first run; set
`BOTSORT_REPO`, `BOTSORT_DET_WEIGHTS` or `BOTSORT_REID_WEIGHTS` to reuse
local copies.

The sample video is downloaded from Wikimedia Commons:
["Pro dog walker - Tokyo - 2024 Nov 1"](https://commons.wikimedia.org/wiki/File:Pro_dog_walker_-_Tokyo_-_2024_Nov_1.webm)
by User:Nesnad, licensed **CC BY-SA 4.0** (commercial use allowed with
attribution; derivatives must carry the same license). The notebook transcodes
it to H.264 MP4 (`dog_walker_tokyo.mp4`) for OpenCV.

The MOT17 dataset is **optional** and not redistributed here. Provide the
MOT17 FRCNN train sequences (set `MOT17_TRAIN`) to enable the MOTA / IDF1
accuracy sections; the video demo, quantization and benchmarks run without it.

Tracking output videos from a notebook run are written to
[artifacts/video/](artifacts/video) (`OV_FP32_tracked.mp4`,
`OV_FP16_tracked.mp4`, `OV_INT8_tracked.mp4`); the notebook clips its video
runs to the first 20 seconds. These files are generated locally and not
committed.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/bot-sort/README.md" />
