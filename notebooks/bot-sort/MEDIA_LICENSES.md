# Media Asset License Audit - bot-sort

Audited: 2026-11-18. Repository policy (`../../CONTRIBUTING.md`): all datasets, images or
videos used inside a notebook must be licensed under Creative Commons with permission for
commercial use.

## Findings

| File | Source | License | Status |
|---|---|---|---|
| `JK_Standing_Next2U.mp4` | Jungkook (BTS) "Standing Next to You" demonstration clip, first 30 s re-encoded at 1920x1080 | **Copyrighted (HYBE/BIGHIT MUSIC)** - no free license | **FAIL** - cannot be redistributed. Also git-tracked. |
| `results/video/OV_FP32_tracked.mp4` | Derived from `JK_Standing_Next2U.mp4` (annotated frames) | Derivative of copyrighted work | **FAIL** - same problem, git-tracked |
| `results/video/OV_FP16_tracked.mp4` | Derived from `JK_Standing_Next2U.mp4` | Derivative of copyrighted work | **FAIL** - git-tracked |
| `results/video/OV_INT8_tracked.mp4` | Derived from `JK_Standing_Next2U.mp4` | Derivative of copyrighted work | **FAIL** - git-tracked |
| Embedded video contact sheets in `bot-sort.ipynb` (FP32/FP16 and INT8 video cells) | Rendered from the JK clip's tracked output | Derivative of copyrighted work | **FAIL** - embedded base64 in the notebook |
| Embedded ReID-fidelity pairs and accuracy/performance chart in `bot-sort.ipynb` | MOT17 crops (not redistributed) and a matplotlib chart | MOT17 CC BY-NC-SA / original | OK - no copyrighted frames |
| `botsort_pipeline_animated.svg` | Original project drawing | Original work | OK (not a third-party asset) |
| `botsort_nb_utils.py` | Original helper code | Original work | OK |

Model weights are **not** redistributed (user-provided via `find_file` / env vars), so they
are out of scope for redistribution. For reference: BoT-SORT code is MIT
(`MultiObjectTrcking/BoT-SORT/LICENSE`); the `bytetrack_x_mot17.pth.tar` and
`mot17_sbs_S50.pth` checkpoints are trained on MOT17, which is CC BY-NC-SA 4.0
(non-commercial). MOT17 sequences are likewise not redistributed here.

## Resolution (PR #3797 review fixes, 2026-11-18)

- `JK_Standing_Next2U.mp4`, `botsort_pipeline_animated.svg` and the three
  `results/video/OV_*_tracked.mp4` derivatives were removed from the repository
  (`git rm`) and added to `.gitignore` so they cannot be committed again.
- The notebook now downloads `dog_walker_tokyo.mp4` at runtime (see below) and
  writes its tracked outputs to `results/video/` (gitignored).
- The notebook no longer embeds copyrighted frames: all cell outputs were
  stripped and the `imageUrl` metadata pointing at the SVG was removed.
- The README no longer references the architecture SVG or the JK clip; it
  documents the free-license video source and license.

## Free-license replacement (verified working)

`dog_walker_tokyo.mp4` - Wikimedia Commons
[File:Pro dog walker - Tokyo - 2024 Nov 1.webm](https://commons.wikimedia.org/wiki/File:Pro_dog_walker_-_Tokyo_-_2024_Nov_1.webm)
by User:Nesnad, **CC BY-SA 4.0** (free license; Commons has no MIT-style video licenses,
CC0/CC BY/CC BY-SA are the free equivalents).

- Original: VP9 webm, 1920x1080, 30 fps, 9.06 s, 2.7 MB.
- Transcoded for OpenCV/`imageio_ffmpeg`: H.264 High, yuv420p, `-movflags +faststart`,
  audio removed -> 7.4 MB.
- Attribution required by CC BY-SA 4.0; any redistributed derivative (including tracked
  outputs) must carry the same CC BY-SA 4.0 license.

Verified end to end with the real pipeline (YOLOX-X detector + FastReID SBS-S50 + BoT-SORT
tracker, FP16 IR on CPU) via `verify_free_license_video.py`:

```
video: dog_walker_tokyo.mp4 | frames: 120 | unique_ids: 22
mean_tracks_per_frame: 9.85 | detector 74.8 ms + ReID 28.5 ms per frame
pipeline_fps: 6.26 | gmc_failures: 0
output: results/video/free_license_check_tracked.mp4
```

The clip is a busy Tokyo sidewalk with multiple pedestrians, so it exercises ID
association, ReID crops and ORB camera-motion compensation the same way the original clip
did. Reproduce with:

```bash
/home/ubuntu/ai-dev/bin/python verify_free_license_video.py
```

## Required follow-up to make the folder compliant

1. Remove `JK_Standing_Next2U.mp4` and the three `results/video/OV_*_tracked.mp4`
   derivatives from the repository (they are git-tracked, so removal must be committed).
2. Point `VIDEO_1` at the free-license clip: rename `dog_walker_tokyo.mp4` to the
   notebook's expected name or set `VIDEO_1=/path/to/dog_walker_tokyo.mp4`, and update
   `botsort_nb_utils.py:82` plus the README and the markdown cell that names the file.
3. Re-run the notebook so the embedded JPEG outputs no longer contain frames from the
   copyrighted clip. This needs the `artifacts/` directory produced by the companion
   PyTorch notebook, which is not present in this checkout.
