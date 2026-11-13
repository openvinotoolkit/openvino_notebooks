#!/usr/bin/env python
"""Regenerate the notebook's evaluation packs (``artifacts/*.npz``) instead of shipping them in git.

``fasterWAM.ipynb`` starts from six recorded packs under ``artifacts/``.  They are generated data
(tens of megabytes), so they are not committed; this script rebuilds them, and
``fwam_ov.ensure_eval_artifacts`` is the notebook's one-line wrapper around it (the notebook runs
this script as a subprocess, so the kernel still never imports torch).

The contract (what ``fwam_ov`` and ``torch_ref_actions.py`` read)::

    eval_samples.npz    image [N,H,W,3] u8, image_norm same f32, state_raw [N,8] f32,
                        state_norm [N,8] f32, gt_action_raw [N,32,7] f32, gt_action_norm [N,32,7]
    eval_samples.json   {"prompts": [...], "samples": [{"task": ...}, ...], ...} - aligned by index
    eval_contexts.npz   context [N,128,4096] f32 (umT5, zero past mask.sum()), mask [N,128] f32 = 1
    eval_latents.npz    lat_video [1,1,48,3,14,28] f32, lat_action [1,1,32,7] f32 (recorded noise)
    eval_proprio.npz    weight [4096,8] f32, bias [4096] f32 (proprio_encoder, the 129th token)
    ov_runtime_pack.npz image [1,3,224,448], lat_video [1,48,3,14,28], lat_action [1,32,7],
                        ctx_p [1,129,4096], ctx_pm [1,129], t_sched [steps], d_sched [steps],
                        action_ref [32,7] (the fp32 torch reference chunk)

Five idempotent steps, each of which verifies what it wrote::

    samples   the released LIBERO LeRobot v3.0 episodes -> the 32-sample evaluation set
              (the only step that needs external data: HF ``yuanty/LIBERO-fastwam``)
    proprio   ``proprio_encoder`` out of the released checkpoint
    contexts  umT5-xxl encode of the 32 prompts (the same loader path ``export_t5.py`` uses)
    latents   the recorded diffusion noise, re-drawn from its seed
    pack      the one-sample runtime regression pack, assembled from the four packs above

Recipes, recovered empirically against the shipped files (``--compare-dir artifacts``):

* **latents** - two *independent* ``torch.Generator("cpu")`` streams, both seeded with the run's
  seed (42): a fresh generator draws ``randn(1,48,3,14,28)``, and a second fresh generator draws
  ``randn(1,32,7)``.  They are not two draws from one generator (that gives a different video
  noise) and numpy's ``default_rng(42)`` gives something else again.  Saved with one extra leading
  axis, hence ``[1,1,48,3,14,28]`` / ``[1,1,32,7]``.  Bit-identical to ``artifacts/eval_latents.npz``.
* **proprio** - the checkpoint is ``{"mot": ..., "step": ..., "torch_dtype": ..., "proprio_encoder":
  {"weight": [4096,8] bf16, "bias": [4096] bf16}}``; the linear layer is stored as-is (no
  transpose), cast to float32.  Bit-identical to ``artifacts/eval_proprio.npz``.
* **contexts** - ``FastWAM.encode_prompt`` semantics: umT5-xxl in **fp32 on the CPU**, zero every
  position past ``mask.sum()``, and return an **all-ones** mask (``encode_prompt`` replaces the
  padding mask with ``ones_like``).  Bit-exactness needs the thread count of the original run:
  ``torch.set_num_threads(24)`` + ``OMP_NUM_THREADS=MKL_NUM_THREADS=24`` reproduces all 32 rows
  bit-for-bit, while 16 threads differs by up to 6.1e-05 on 12 of the 32 prompts (MKL reduction
  order).  ``--threads`` defaults to 24 for that reason.
* **pack** - ``ctx_p`` = the 129-token stream, 128 *zero* text tokens (an empty instruction, as in
  the recorded pack: the pack checks the three graphs, not the text path) plus the proprio token
  ``state_norm @ W.T + b``; ``ctx_pm`` = all ones, ``t_sched``/``d_sched`` =
  ``fwam_ov.build_schedule(steps)``, and
  ``action_ref`` = the fp32 torch reference chunk **for this pack's own inputs**, which is why the
  pack step runs ``torch_ref_actions.py`` once when no matching ``torch_ref_pack.npz`` is on disk
  (``--torch-ref-dtype none`` skips that and writes the pack without a reference).

The shipped ``ov_runtime_pack.npz`` is *not* eval sample 0: it is the exporter's export-time
reference input (``utils/fwam_export/artifacts/ref_inputs_libero_fasterwam_2cam224_1e-4.pt``, which
is where its ``image`` comes from byte-for-byte), so a regenerated pack is a *different sample* by
construction - compare it structurally, not array-by-array.  The shipped pack additionally carries
``first_frame``, ``kv00..kv15`` and ``action_src`` from the original fp16 OpenVINO probe; those are
diagnostics for ``torch_ref_actions.py``'s self-checks, not inputs of the notebook, and are not
reproduced here.

Usage (from the notebook folder, with the torch venv)::

    .venv/bin/python utils/fwam_export/make_eval_packs.py --steps all
    .venv/bin/python utils/fwam_export/make_eval_packs.py --steps proprio,latents --compare-dir artifacts
    .venv/bin/python utils/fwam_export/make_eval_packs.py --steps samples   # needs the episode data

Nothing here needs OpenVINO, NNCF or the GPU: this is export-time bookkeeping, not inference.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # <notebook>/utils/fwam_export
NOTEBOOK = HERE.parents[1]  # <notebook>
TORCH_REF_SCRIPT = NOTEBOOK / "torch_ref_actions.py"

STEPS = ("samples", "proprio", "contexts", "latents", "pack")

# The 32-sample LIBERO evaluation set: 8 samples from each of the four released suites, taken
# mid-episode with a 32-step ground-truth horizon - the same draw `scripts/ov_eval_data.py` makes
# (per-suite 8, horizon 32, seed 42), so the packs stay comparable with the project's own runs.
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PER_SUITE = 8
HORIZON = 32
SAMPLE_SEED = 42
RAW_DATA_REL = "data/libero_mujoco3.3.2_raw/lerobot_v30"

# Where the observations come from: the released episodes, already unpacked as a LeRobot v3.0 tree.
# Only that tree is fetched - the repo's `*.tar.gz` copies are the same bytes again (9.4 GB in total,
# 4.7 GB for the tree), so downloading them would double the transfer for nothing.
DATA_HF_REPO = "yuanty/LIBERO-fastwam"
DATA_HF_PATTERNS = ("lerobot_v30/*",)
DATA_HF_IGNORE = ("*.tar.gz",)
DATA_HF_GB = 4.7

# umT5-xxl context: 128 tokens x 4096, the width the video and action graphs take (before the
# proprio token is appended), and the CPU thread count that reproduces the shipped bank exactly.
CTX_SEQ = 128
CTX_DIM = 4096
T5_THREADS = 24

# The recorded diffusion noise: the released checkpoint's latent shapes and the run's seed.  See the
# module docstring for the convention (two independent generators, both seeded with NOISE_SEED).
NOISE_SEED = 42
VIDEO_LATENT_SHAPE = (1, 48, 3, 14, 28)
ACTION_LATENT_SHAPE = (1, 32, 7)

PACK_STEPS = 10
# Diagnostics `torch_ref_actions.py` self-checks the pack against: the first latent frame the VAE
# encoder produces, and the 16 fused K/V tensors of the video prefill (8 tapped blocks x k, v).
FIRST_FRAME_SHAPE = (1, 48, 1, 14, 28)
VIDEO_KV_SHAPE = (1, 294, 3072)


def log(step: str, *parts: str) -> None:
    """Print one ``[step] message`` line, flushed (the notebook streams this)."""
    print(f"[{step}]", *parts, flush=True)


# --------------------------------------------------------------------------- small helpers


def _load(path: Path) -> dict:
    """Load an ``.npz`` into a plain dict of arrays (closes the file)."""
    with np.load(path) as z:
        return {k: np.array(z[k]) for k in z.files}


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 1) if path.is_file() else 0.0


def _verify(path: Path, expect: dict[str, tuple[tuple[int, ...], str]]) -> dict:
    """Re-open a pack and check that every expected array is there with the right shape and dtype.

    Args:
        path: the ``.npz`` just written.
        expect: ``{key: (shape, dtype)}`` that must be present exactly.

    Returns:
        The loaded arrays.

    Raises:
        RuntimeError: when the file is unreadable, or a key is missing / wrong.
    """
    if not path.is_file():
        raise RuntimeError(f"{path} was not written")
    got = _load(path)
    problems = []
    for key, (shape, dtype) in expect.items():
        if key not in got:
            problems.append(f"{key}: missing")
            continue
        array = got[key]
        if tuple(array.shape) != tuple(shape):
            problems.append(f"{key}: shape {tuple(array.shape)} != {tuple(shape)}")
        if array.dtype != np.dtype(dtype):
            problems.append(f"{key}: dtype {array.dtype} != {np.dtype(dtype)}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            problems.append(f"{key}: not finite")
    if problems:
        raise RuntimeError(f"{path} is not the pack this script promised: " + "; ".join(problems))
    log(path.name.split(".")[0], f"verified {path.name} ({_size_mb(path)} MB, {len(got)} arrays)")
    return got


def _compare(path: Path, compare_dir: Path | None) -> None:
    """Report, per array, how a freshly written pack compares with a reference copy (if any)."""
    if not compare_dir:
        return
    if path.suffix != ".npz":  # the JSON sidecar, which numpy cannot load
        log("compare", f"{path.name}: JSON sidecar, compared by key below only where it is .npz")
        return
    ref_path = Path(compare_dir) / path.name
    if not ref_path.is_file():
        log("compare", f"no {ref_path} - nothing to compare against")
        return
    got, ref = _load(path), _load(ref_path)
    log("compare", f"{path.name} vs {ref_path}")
    for key in sorted(set(ref) & set(got)):
        a, b = np.asarray(ref[key]), np.asarray(got[key])
        if a.shape != b.shape:
            log("compare", f"  {key:<14} shape {b.shape} != reference {a.shape} (different sample)")
        elif a.dtype.kind == "f":
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            same = np.array_equal(a, b)
            log("compare", f"  {key:<14} {'bit-identical' if same else 'differs'} " f"max|d|={delta.max():.3e} mean|d|={delta.mean():.3e}")
        else:
            log("compare", f"  {key:<14} {'bit-identical' if np.array_equal(a, b) else 'differs'}")


def _have(path: Path, keys: list[str]) -> bool:
    """True when ``path`` exists and holds every one of ``keys`` (a usable cache hit)."""
    if not path.is_file():
        return False
    try:
        with np.load(path) as z:
            return all(k in z.files for k in keys)
    except (OSError, ValueError):
        return False


_THREADS: int | None = None


def _torch():
    """Import torch lazily (after the thread environment is set) and pin its thread count."""
    import torch  # noqa: PLC0415  (deliberately late; see set_threads)

    if _THREADS:
        torch.set_num_threads(int(_THREADS))
    return torch


def set_threads(threads: int) -> int:
    """Pin torch/MKL/OMP to ``threads`` and return the value used.

    ``torch.randn`` is thread-count independent, but the fp32 fp GEMMs the text encoder runs are
    not: 24 threads reproduces ``artifacts/eval_contexts.npz`` bit-for-bit on a 16-core host, 16
    threads differs in the 5th decimal on 12 of 32 prompts.  Must be called before ``import torch``
    for the ``OMP``/``MKL`` variables to be honoured.
    """
    global _THREADS
    threads = int(threads or T5_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    _THREADS = threads
    return threads


# --------------------------------------------------------------------------- 1. samples


def _fetch_eval_data(staging: Path) -> None:
    """Download the released LIBERO episodes into ``staging`` (which then holds ``lerobot_v30/``).

    This is the only thing this generator cannot compute: 4.7 GB of packed mp4s and parquet from
    ``yuanty/LIBERO-fastwam``.  ``snapshot_download`` resumes, so an interrupted fetch is picked up
    where it stopped, and it prints its own progress bar through to the notebook cell.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415  (only the samples step needs it)

    staging.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATA_HF_REPO,
        repo_type="dataset",
        local_dir=str(staging),
        allow_patterns=list(DATA_HF_PATTERNS),
        ignore_patterns=list(DATA_HF_IGNORE),
    )


def _samples_fail(art_dir: Path, data_root: Path, why: str) -> SystemExit:
    """Build the one actionable error the ``samples`` step raises when it cannot run."""
    return SystemExit(
        f"[samples] {why}\n"
        f"[samples] this is the only step that needs external data, and the notebook fetches it for\n"
        f"[samples] you: Hugging Face `{DATA_HF_REPO}` (repo_type=dataset), the already-unpacked\n"
        f"[samples] `lerobot_v30/` tree only - {DATA_HF_GB} GB of the repo's 9.4 GB, the tarballs are\n"
        f"[samples] skipped.  It lands in {data_root}\n"
        f"[samples]   (which must hold {', '.join(SUITES)}_no_noops_lerobot/)\n"
        f"[samples] fetch it with:  huggingface-cli download {DATA_HF_REPO} --repo-type dataset \\\n"
        f"[samples]                    --include 'lerobot_v30/*' --exclude '*.tar.gz' \\\n"
        f"[samples]                    --local-dir {data_root.parent}\n"
        f"[samples] or pass --data-root <a copy you already have>, or --no-download to skip trying.\n"
        f"[samples] reading it also needs `av`, `pandas` and `pyarrow` for the mp4s and parquet, plus\n"
        f"[samples] `gitpython` and `torchvision` for the repository's own import chain - the notebook\n"
        f"[samples] installs all five in its setup cell.\n"
        f"[samples] fetch it, or pass --data-root, and re-run:  --steps samples\n"
        f"[samples] the other steps (proprio, contexts, latents, pack) need no download beyond the\n"
        f"[samples] checkpoint and the umT5 weights the notebook already uses."
    )


class VideoReader:
    """Decode one frame of a LeRobot v3.0 packed mp4 (many episodes per file)."""

    def __init__(self) -> None:
        self.containers: dict[str, object] = {}
        self.streams: dict[str, object] = {}

    def get(self, mp4: str, ts: float, tol: float = 0.03) -> np.ndarray:
        """Frame at (or the last one before) timestamp ``ts`` of a packed mp4."""
        import av  # noqa: PLC0415

        container = self.containers.get(mp4)
        if container is None:
            container = av.open(mp4)
            self.containers[mp4] = container
            self.streams[mp4] = container.streams.video[0]
        stream = self.streams[mp4]
        time_base = float(stream.time_base)
        container.seek(int(ts / time_base), stream=stream)
        last = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * time_base)
            if pts >= ts - tol:
                return frame.to_ndarray(format="rgb24")
            last = frame
        if last is not None and last.pts is not None and abs(float(last.pts * time_base) - ts) < 1.0:
            return last.to_ndarray(format="rgb24")
        raise RuntimeError(f"no frame at ts={ts} in {mp4}")


def center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Copy of ``eval_libero_single._center_crop_resize`` (PIL bilinear, uint8 in/out)."""
    from PIL import Image  # noqa: PLC0415

    pil = Image.fromarray(image)
    src_w, src_h = pil.size
    scale = max(width / src_w, height / src_h)
    pil = pil.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = pil.size
    left, top = max((rw - width) // 2, 0), max((rh - height) // 2, 0)
    return np.asarray(pil.crop((left, top, left + width, top + height)), dtype=np.uint8)


def load_suite(root: Path, suite: str):
    """Rows, episodes, task strings and fps of one LeRobot v3.0 suite."""
    import pandas as pd  # noqa: PLC0415

    base = root / f"{suite}_no_noops_lerobot"
    rows = pd.read_parquet(str(base / "data/chunk-000/file-000.parquet"))
    episodes = pd.read_parquet(str(base / "meta/episodes/chunk-000/file-000.parquet"))
    tasks_raw = pd.read_parquet(str(base / "meta/tasks.parquet"))
    fps = float(json.loads((base / "meta/info.json").read_text())["fps"])
    tasks = (
        {int(task_index): str(name) for name, task_index in tasks_raw["task_index"].items()}
        if "task_index" in tasks_raw.columns
        else {int(i): str(n) for i, n in enumerate(tasks_raw.index)}
    )
    return rows, episodes, tasks, fps


def pick_samples(root: Path, suites, per_suite: int, horizon: int, seed: int) -> list[dict]:
    """Deterministic spread: round-robin over tasks x episodes, mid-episode observation frame.

    This is ``scripts/ov_eval_data.py:pick_samples`` (LIBERO path) unchanged in behaviour, so the
    packs it produces line up with the project's recorded evaluation numbers.
    """
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for suite in suites:
        rows, episodes, tasks, fps = load_suite(root, suite)
        state = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(rows["action"].to_numpy()).astype(np.float32)
        lengths = episodes["length"].to_numpy()
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        by_task: dict[int, list[int]] = {}
        for ep, _index in enumerate(episodes["episode_index"].to_numpy()):
            by_task.setdefault(int(rows["task_index"].to_numpy()[starts[ep]]), []).append(ep)
        tasks_sorted = sorted(by_task)
        picked: list[tuple[int, int, int]] = []
        round_i = 0
        while len(picked) < per_suite and tasks_sorted:
            progressed = False
            for task_index in tasks_sorted:
                if len(picked) >= per_suite:
                    break
                eps = by_task[task_index]
                if round_i >= len(eps):
                    continue
                if int(lengths[eps[round_i]]) <= horizon + 4:
                    continue
                offset = int(rng.integers(2, int(lengths[eps[round_i]]) - horizon - 1))
                picked.append((eps[round_i], offset, task_index))
                progressed = True
            round_i += 1
            if not progressed:
                break
        for ep, offset, task_index in picked:
            i = int(starts[ep]) + offset
            if i + horizon > int(starts[ep]) + int(lengths[ep]):
                continue
            row = episodes.iloc[ep]
            video_base = root / f"{suite}_no_noops_lerobot" / "videos"

            def cam(name: str, row=row, video_base=video_base) -> tuple[str, float]:
                key = f"videos/observation.images.{name}"
                return (
                    str(
                        video_base
                        / f"observation.images.{name}"
                        / f"chunk-{int(row[key + '/chunk_index']):03d}"
                        / f"file-{int(row[key + '/file_index']):03d}.mp4"
                    ),
                    float(row[key + "/from_timestamp"]),
                )

            image_mp4, image_from = cam("image")
            wrist_mp4, wrist_from = cam("wrist_image")
            out.append(
                {
                    "suite": suite,
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(offset),
                    "task_index": task_index,
                    "task": tasks.get(task_index, ""),
                    "row": int(i),
                    "image_mp4": image_mp4,
                    "wrist_mp4": wrist_mp4,
                    "ts": image_from + offset / fps,
                    "ts_wrist": wrist_from + offset / fps,
                    "state_raw": state[i],
                    "gt_action_raw": action[i : i + horizon],
                }
            )
    return out


def _repo_default_prompt(clone: Path) -> str:
    """Return the repo's ``DEFAULT_PROMPT`` template, read out of its source file.

    Importing it from ``fasterwam.datasets.lerobot.robot_video_dataset`` would drag in torchvision,
    Hugging Face ``datasets`` and the LeRobot reader - the whole training stack - for one string
    constant, and this environment deliberately installs only what the export and the CPU reference
    need.  The literal is parsed from the clone instead, so it still cannot drift from the repo.
    """
    path = clone / "src" / "fasterwam" / "datasets" / "lerobot" / "robot_video_dataset.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEFAULT_PROMPT" for t in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str) or "{task}" not in value:
                raise RuntimeError(f"DEFAULT_PROMPT in {path} is not a '{{task}}' template: {value!r}")
            return value
    raise RuntimeError(f"no DEFAULT_PROMPT assignment in {path}")


def step_samples(cfg) -> dict:
    """Build ``eval_samples.npz`` + ``eval_samples.json`` from the released LIBERO episodes.

    What one sample is (the same thing the LIBERO eval harness builds from a simulator step):

    * ``image``: primary camera | wrist camera, each 512x512 -> center-crop-resize 224x224 ->
      horizontal concat -> ``[224,448,3]`` uint8, and ``image_norm = image * 2/255 - 1``
    * ``state_raw`` / ``gt_action_raw``: the dataset's ``observation.state`` / 32 rows of ``action``
    * ``state_norm`` / ``gt_action_norm``: the checkpoint's own min/max normalizer
      (``dataset_stats.json`` through ``FastWAMProcessor``), i.e. the image the graphs see
    * one prompt per sample: ``DEFAULT_PROMPT.format(task=<instruction>)``
    """
    art_dir, clone = cfg.art_dir, cfg.clone
    data_root = Path(cfg.data_root) if cfg.data_root else clone / RAW_DATA_REL
    npz_path = art_dir / "eval_samples.npz"
    json_path = art_dir / "eval_samples.json"
    if (
        not cfg.force
        and _have(npz_path, ["image", "image_norm", "state_norm", "gt_action_norm"])
        and json_path.is_file()
        and len(json.loads(json_path.read_text()).get("samples", [])) == PER_SUITE * len(SUITES)
    ):
        log("samples", f"reusing {npz_path.name} + {json_path.name} ({_size_mb(npz_path)} MB)")
        return {"step": "samples", "status": "cache", "paths": [npz_path, json_path], "seconds": 0.0}

    # Check the readers before fetching 4.7 GB: if this interpreter cannot decode the episodes, the
    # download would be wasted.
    try:
        import av  # noqa: F401,PLC0415  (packed-mp4 decoding)
        import pandas  # noqa: F401,PLC0415  (parquet)
    except ImportError as exc:
        raise _samples_fail(art_dir, data_root, f"this interpreter cannot read the episodes ({exc})")

    missing_dirs = [s for s in SUITES if not (data_root / f"{s}_no_noops_lerobot").is_dir()]
    if not data_root.is_dir() or missing_dirs:
        # The one download in this notebook, and it only happens on the default path: an explicit
        # --data-root or --no-download means "this is my copy, leave it alone".
        if cfg.download and not cfg.data_root:
            log(
                "samples",
                f"the LIBERO episodes are not under {data_root} - fetching "
                f"{DATA_HF_REPO} from Hugging Face ({DATA_HF_GB} GB of packed mp4s and "
                f"parquet, the unpacked lerobot_v30 tree; the tarballs are skipped)",
            )
            _fetch_eval_data(data_root.parent)
            missing_dirs = [s for s in SUITES if not (data_root / f"{s}_no_noops_lerobot").is_dir()]
        if not data_root.is_dir() or missing_dirs:
            raise _samples_fail(
                art_dir,
                data_root,
                f"the released LIBERO eval data is not under {data_root}" + (f" (missing {', '.join(missing_dirs)})" if missing_dirs else ""),
            )

    # The repo helpers are imported from the clone, not from this notebook: `run_infer_smoke.py`
    # imports `experiments/latency/measure_latency.py` from `$REPO` (which it defaults to another
    # machine's path), and both it and the `fasterwam` package below come from the clone.  So point
    # REPO at this clone and put the two import roots on sys.path *before* importing them.
    os.environ["REPO"] = str(clone)
    for extra in (clone / "src", clone / "experiments" / "latency", HERE):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    # Everything below mirrors scripts/ov_eval_data.py: same config, same normalizer, same draw.
    from hydra.utils import instantiate  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    import run_infer_smoke as RIS  # noqa: PLC0415  (the repo's model-build recipe)

    default_prompt = _repo_default_prompt(clone)
    from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: PLC0415

    torch = _torch()
    torch.set_num_threads(4)
    stats = clone / cfg.stats_rel
    log("samples", f"{len(SUITES)} suites x {PER_SUITE} samples, horizon {HORIZON}, " f"seed {SAMPLE_SEED}, data {data_root}")
    model_cfg = RIS.load_cfg("fasterwam", None)
    OmegaConf.set_struct(model_cfg, False)
    OmegaConf.set_readonly(model_cfg, False)
    processor = instantiate(model_cfg.data.train.processor)
    processor.eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
    img_meta = [meta["shape"] for meta in processor.shape_meta["images"]]
    height, width = int(img_meta[0][1]), int(img_meta[0][2])
    concat = str(model_cfg.data.train.get("concat_multi_camera", "horizontal"))

    def normalise(state, action=None):
        """Mirror the eval harness's ``_normalize_proprio`` / ``_denormalize_action``."""
        state_key = list(processor.shape_meta["state"])[0]["key"]
        action_key = list(processor.shape_meta["action"])[0]["key"]
        batch = {"state": {state_key: state}}
        if action is not None:
            batch["action"] = {action_key: action}
        batch = processor.normalizer.forward(processor.action_state_transform(batch))
        return batch["state"][state_key], (batch["action"][action_key] if action is not None else None)

    t_start = time.perf_counter()
    t0 = time.perf_counter()
    samples = pick_samples(data_root, SUITES, PER_SUITE, HORIZON, SAMPLE_SEED)
    log("samples", f"selected {len(samples)} samples in {time.perf_counter() - t0:.1f} s")
    reader = VideoReader()
    images, images_norm, states, states_norm, gt_raw, gt_norm = [], [], [], [], [], []
    prompts, meta = [], []
    decode_s = 0.0
    for index, sample in enumerate(samples):
        t0 = time.perf_counter()
        primary = center_crop_resize(reader.get(sample["image_mp4"], sample["ts"]), width, height)
        wrist = center_crop_resize(reader.get(sample["wrist_mp4"], sample["ts_wrist"]), width, height)
        rgb = np.concatenate([primary, wrist], axis=1) if concat == "horizontal" else np.concatenate([primary, wrist], axis=0)
        decode_s += time.perf_counter() - t0
        state = torch.as_tensor(sample["state_raw"][None], dtype=torch.float32)
        action = torch.as_tensor(sample["gt_action_raw"], dtype=torch.float32)
        images.append(rgb)
        images_norm.append(rgb.astype(np.float32) * (2.0 / 255.0) - 1.0)
        states.append(sample["state_raw"])
        states_norm.append(normalise(state)[0][0].numpy())
        gt_raw.append(sample["gt_action_raw"])
        gt_norm.append(normalise(state, action)[1].numpy())
        prompts.append(default_prompt.format(task=sample["task"]))
        meta.append(
            {
                k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                for k, v in sample.items()
                if k not in ("state_raw", "gt_action_raw")
            }
        )
        if index % 8 == 0:
            log("samples", f"  {index + 1}/{len(samples)} {sample['suite']} " f"ep{sample['episode_index']} f{sample['frame_index']} " f"ts={sample['ts']:.2f}")
    images = np.stack(images)
    images_norm = np.stack(images_norm)
    expected_hw = (2 * height, width) if concat == "vertical" else (height, 2 * width)
    if tuple(images.shape[1:3]) != tuple(expected_hw):
        raise RuntimeError(f"assembled a {images.shape[1:3]} mosaic, the model wants {expected_hw}")
    np.savez_compressed(
        npz_path,
        image=images.astype(np.uint8),
        image_norm=images_norm.astype(np.float32),
        state_raw=np.stack(states).astype(np.float32),
        state_norm=np.stack(states_norm).astype(np.float32),
        gt_action_raw=np.stack(gt_raw).astype(np.float32),
        gt_action_norm=np.stack(gt_norm).astype(np.float32),
    )
    json_path.write_text(
        json.dumps(
            {
                "prompts": prompts,
                "samples": meta,
                "dataset": "libero",
                "task": None,
                "image_hw": list(images.shape[1:3]),
                "horizon": HORIZON,
                "suites": list(SUITES),
                "per_suite": PER_SUITE,
                "seed": SAMPLE_SEED,
                "decode_seconds": round(decode_s, 2),
            },
            indent=1,
        )
    )
    n = len(prompts)
    _verify(
        npz_path,
        {
            "image": ((n, expected_hw[0], expected_hw[1], 3), "uint8"),
            "image_norm": ((n, expected_hw[0], expected_hw[1], 3), "float32"),
            "state_norm": ((n, 8), "float32"),
            "gt_action_norm": ((n, HORIZON, 7), "float32"),
        },
    )
    log("samples", f"wrote {json_path.name} ({n} prompts, {_size_mb(json_path)} MB); " f"image decode {decode_s:.1f} s")
    return {"step": "samples", "status": "written", "paths": [npz_path, json_path], "seconds": round(time.perf_counter() - t_start, 1)}


# --------------------------------------------------------------------------- 2. proprio


def step_proprio(cfg) -> dict:
    """Write ``eval_proprio.npz`` from the checkpoint's ``proprio_encoder`` linear layer.

    ``FastWAM._append_proprio_to_context`` turns the 8-d normalised state into the 129th context
    token with ``proprio_encoder`` = ``nn.Linear(8, 4096)``.  The checkpoint stores it as its own
    top-level entry (``{"mot": ..., "step": ..., "proprio_encoder": {"weight", "bias"}}``), in
    bfloat16, weight already ``[4096, 8]`` - no transpose, only a cast to float32.

    The checkpoint is mapped, not read: ``torch.load(..., mmap=True)`` keeps the 11 GB file out of
    RSS, so this step costs a few megabytes rather than a full checkpoint load.
    """
    out = cfg.art_dir / "eval_proprio.npz"
    if not cfg.force and _have(out, ["weight", "bias"]):
        log("proprio", f"reusing {out.name} ({_size_mb(out)} MB)")
        return {"step": "proprio", "status": "cache", "paths": [out], "seconds": 0.0}
    if not cfg.ckpt.is_file():
        raise RuntimeError(f"missing the released checkpoint at {cfg.ckpt} - " f"run fwam_ov.download_checkpoint() first")
    torch = _torch()
    t0 = time.perf_counter()
    state = torch.load(str(cfg.ckpt), map_location="cpu", mmap=True, weights_only=True)
    layer = state.get("proprio_encoder") if isinstance(state, dict) else None
    if layer is None:
        # Some checkpoints nest the state dict one level down; look for the projection by name.
        for key, value in (state or {}).items():
            if "proprio" in str(key) and hasattr(value, "keys"):
                layer = value
                break
    if layer is None or "weight" not in layer:
        raise RuntimeError(f"{cfg.ckpt} has no 'proprio_encoder' linear layer " f"(top-level keys: {list(state)[:8]})")
    weight = layer["weight"].detach().float().numpy().astype(np.float32)
    bias = layer["bias"].detach().float().numpy().astype(np.float32)
    if weight.shape != (CTX_DIM, 8) or bias.shape != (CTX_DIM,):
        raise RuntimeError(f"proprio_encoder is {weight.shape}/{bias.shape}, " f"the context token needs ({CTX_DIM}, 8)/({CTX_DIM},)")
    np.savez_compressed(out, weight=weight, bias=bias)
    _verify(out, {"weight": ((CTX_DIM, 8), "float32"), "bias": ((CTX_DIM,), "float32")})
    log("proprio", f"wrote {out.name} from {cfg.ckpt.name} " f"({layer['weight'].dtype} -> float32) in {time.perf_counter() - t0:.1f} s")
    return {"step": "proprio", "status": "written", "paths": [out], "seconds": round(time.perf_counter() - t0, 1)}


# --------------------------------------------------------------------------- 3. contexts


def _notebook_on_sys_path(cfg) -> None:
    """Make ``export_t5`` importable and point it at this notebook's clone (it reads env at import)."""
    os.environ.setdefault("REPO", str(cfg.clone))
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(cfg.clone / "checkpoints"))
    for path in (str(HERE), str(cfg.clone / "src"), str(cfg.clone / "experiments" / "latency")):
        if path not in sys.path:
            sys.path.insert(0, path)


def _mmap_text_encoder(cfg) -> None:
    """Make the repository's text-encoder load map the 11.4 GB file instead of reading + widening it.

    ``_load_registered_model`` calls ``load_state_dict(path, torch_dtype=float32)``, which holds the
    whole checkpoint *and* the fp32 model at once (~38 GB, OOM on a 16-core laptop).  Mapping it and
    letting ``load_state_dict`` cast per parameter keeps the peak at the fp32 model alone (~19 GB).
    """
    import torch  # noqa: PLC0415

    from fasterwam.models.wan22.helpers import loader as repo_loader  # noqa: PLC0415

    def mapped(path, torch_dtype=None, device="cpu"):
        state = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
        if len(state) == 1:  # {"state_dict": ...} style wrappers
            for key in ("state_dict", "module", "model_state"):
                if key in state:
                    state = state[key]
                    break
        return state

    repo_loader.load_state_dict = mapped


def step_contexts(cfg) -> dict:
    """Write ``eval_contexts.npz``: the umT5-xxl context of the 32 evaluation prompts.

    The loader is ``utils/fwam_export/export_t5.py``'s (the same one the text-encoder export gates
    with), because that is the path already validated against this file: fp32 weights and
    activations on the CPU, the ``T5Enc`` wrapper that reproduces ``FastWAM.encode_prompt``, and
    the ``encode_prompt`` post-processing the notebook's contract states - every position past
    ``mask.sum()`` zeroed, and an **all-ones** mask stored next to it.

    Prompts come from ``eval_samples.json`` (``prompts``, one per sample); when that sidecar is
    missing the sample ``task`` strings are wrapped in ``fwam_ov.t5_prompt`` instead.
    """
    out = cfg.art_dir / "eval_contexts.npz"
    samples_json = cfg.art_dir / "eval_samples.json"
    if not cfg.force and _have(out, ["context", "mask"]):
        log("contexts", f"reusing {out.name} ({_size_mb(out)} MB)")
        return {"step": "contexts", "status": "cache", "paths": [out], "seconds": 0.0}
    if not samples_json.is_file():
        raise RuntimeError(f"missing {samples_json} - the contexts step encodes its prompts " f"(run --steps samples first)")
    sidecar = json.loads(samples_json.read_text())
    prompts = sidecar.get("prompts") or [s["task"] for s in sidecar["samples"]]
    _notebook_on_sys_path(cfg)
    _mmap_text_encoder(cfg)
    torch = _torch()  # pins torch to cfg.threads before any GEMM runs
    log("contexts", f"{len(prompts)} prompts, umT5-xxl fp32 on the CPU, " f"{torch.get_num_threads()} threads (that is what makes this reproducible)")

    import export_t5 as ET  # noqa: PLC0415  (loader + wrapper, shared with the T5 export)
    import run_infer_smoke as RIS  # noqa: PLC0415

    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    model_cfg = RIS.load_cfg("fasterwam", None)
    text_encoder, tokenizer, ckpt = ET.build_text_encoder(model_cfg)
    log("contexts", f"built the encoder in {time.perf_counter() - t0:.1f} s " f"({sum(p.numel() for p in text_encoder.parameters()) / 1e9:.2f} B params)")
    encoder = ET.T5Enc(text_encoder).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = np.asarray(ids, dtype=np.int64)
    token_mask = np.asarray(mask, dtype=np.int64)
    if ids.shape[1] != CTX_SEQ:
        raise RuntimeError(f"the tokenizer produced {ids.shape[1]} tokens, the graphs take {CTX_SEQ}")
    context = np.zeros((len(prompts), CTX_SEQ, CTX_DIM), np.float32)
    t0 = time.perf_counter()
    for index in range(len(prompts)):
        context[index] = encoder(torch.from_numpy(ids[index : index + 1]), torch.from_numpy(token_mask[index : index + 1])).numpy()
        if index % 8 == 0:
            log("contexts", f"  {index + 1}/{len(prompts)} live tokens " f"{int(token_mask[index].sum())}")
    encoded_s = time.perf_counter() - t0
    log("contexts", f"encoded {len(prompts)} prompts in {encoded_s:.1f} s")
    # encode_prompt returns `torch.ones_like(mask)`: the 129th token is the proprio one and the
    # graphs are fed an all-live mask, the padding is already zeroed inside the context.
    np.savez_compressed(out, context=context, mask=np.ones((len(prompts), CTX_SEQ), np.float32))
    _verify(out, {"context": ((len(prompts), CTX_SEQ, CTX_DIM), "float32"), "mask": ((len(prompts), CTX_SEQ), "float32")})
    log("contexts", f"wrote {out.name} ({_size_mb(out)} MB)")
    return {"step": "contexts", "status": "written", "paths": [out], "seconds": round(encoded_s, 1)}


# --------------------------------------------------------------------------- 4. latents


def step_latents(cfg) -> dict:
    """Write ``eval_latents.npz``: the recorded diffusion noise, re-drawn from its seed.

    The convention (recovered by matching the shipped file bit-exactly, see the module docstring):
    two *independent* ``torch.Generator`` streams, each seeded with the run's seed, drawing the
    video latent ``[1,48,3,14,28]`` and the action latent ``[1,32,7]``.  The file keeps one extra
    leading axis, which is why the stored shapes are ``[1,1,...]``.
    """
    out = cfg.art_dir / "eval_latents.npz"
    if not cfg.force and _have(out, ["lat_video", "lat_action"]):
        log("latents", f"reusing {out.name} ({_size_mb(out)} MB)")
        return {"step": "latents", "status": "cache", "paths": [out], "seconds": 0.0}
    torch = _torch()
    seed = int(cfg.noise_seed)

    def draw(shape: tuple[int, ...]) -> np.ndarray:
        generator = torch.Generator(device="cpu").manual_seed(seed)  # a *fresh* stream each time
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).numpy()

    lat_video = draw(VIDEO_LATENT_SHAPE)
    lat_action = draw(ACTION_LATENT_SHAPE)
    if not np.isfinite(lat_video).all() or not np.isfinite(lat_action).all():
        raise RuntimeError("the noise draw produced non-finite values")
    np.savez_compressed(out, lat_video=lat_video[None], lat_action=lat_action[None])
    _verify(out, {"lat_video": ((1,) + VIDEO_LATENT_SHAPE, "float32"), "lat_action": ((1,) + ACTION_LATENT_SHAPE, "float32")})
    log("latents", f"wrote {out.name} (torch.randn, seed {seed}, " f"independently seeded video/action streams)")
    return {"step": "latents", "status": "written", "paths": [out], "seconds": 0.0}


# --------------------------------------------------------------------------- 5. pack


def _proprio_token(state_norm: np.ndarray, art_dir: Path) -> np.ndarray:
    """``state_norm @ weight.T + bias`` for one sample: the 129th context token, ``[1,4096]``."""
    with np.load(art_dir / "eval_proprio.npz") as z:
        weight, bias = z["weight"], z["bias"]
    return (state_norm[None, :].astype(np.float32) @ weight.T + bias).astype(np.float32)


def _cos(first: np.ndarray, second: np.ndarray) -> float:
    """Cosine similarity of two flattened arrays (recomputed, not copied from a run)."""
    a = np.asarray(first, np.float64).ravel()
    b = np.asarray(second, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    """Root-mean-square error of two arrays."""
    delta = np.asarray(first, np.float64) - np.asarray(second, np.float64)
    return float(np.sqrt((delta * delta).mean()))


def _reference_checks(pack: dict, reference: dict) -> dict:
    """The five self-checks ``torch_ref_actions.py`` reports, recomputed against the finished pack."""
    kv = np.stack([pack[f"kv{i:02d}"].astype(np.float32) for i in range(len(reference["kv"]))])
    return {
        "action_vs_pack_cos": _cos(pack["action_ref"], reference["action"]),
        "action_vs_pack_rmse": _rmse(pack["action_ref"], reference["action"]),
        "first_frame_vs_pack_cos": _cos(pack["first_frame"], reference["first_frame"]),
        "kv_vs_pack_cos": _cos(kv, reference["kv"]),
        "kv_vs_pack_rmse": _rmse(kv, reference["kv"]),
    }


def _content_id(payload: dict) -> str:
    """Hash the pack's *inputs* (arrays, not file bytes) so a cached reference can be matched to them.

    The bytes of an ``.npz`` are not stable (zip timestamps), and the pack cannot hash itself anyway
    - it carries the reference the hash is used to look up - so the inputs are hashed array by array.
    """
    digest = hashlib.sha256()
    for key in sorted(payload):
        array = np.ascontiguousarray(payload[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cached_reference(cfg, content_id: str) -> dict | None:
    """Return the cached torch reference when it was computed from exactly these pack inputs."""
    out = cfg.art_dir / "torch_ref_pack.npz"
    if not out.is_file():
        return None
    with np.load(out) as z:
        if "content_id" not in z.files or str(z["content_id"]) != content_id:
            return None
        return {k: (z[k].item() if z[k].ndim == 0 else np.array(z[k])) for k in z.files}


def _write_reference_cache(cfg, pack: dict, reference: dict, content_id: str) -> Path:
    """Write ``artifacts/torch_ref_pack.npz`` from a bootstrap run.

    The ``checks`` are recomputed against the pack this function's caller has just written - whose
    ``action_ref``/``first_frame``/``kv*`` *are* this run's output, so on a bootstrap run they read
    cos 1.0 by construction.  ``baseline="self"`` records that, so nothing downstream mistakes them
    for an independent check; a later ``torch_ref_actions.py`` run against a pack that already
    carries a reference produces the real comparison.
    """
    out = cfg.art_dir / "torch_ref_pack.npz"
    pack_file = cfg.art_dir / "ov_runtime_pack.npz"
    # `identity.pack_md5` is what `fwam_ov.ensure_torch_ref` compares against the pack on disk; the
    # extra `content_id` is this script's own, input-based key (see _content_id).
    identity = {
        "pack_md5": hashlib.md5(pack_file.read_bytes()).hexdigest() if pack_file.is_file() else None,
        "ckpt_bytes": cfg.ckpt.stat().st_size if cfg.ckpt.is_file() else None,
    }
    checks = _reference_checks(pack, reference)
    np.savez_compressed(
        out,
        action=np.asarray(reference["action"], np.float32),
        first_frame=np.asarray(reference["first_frame"], np.float32),
        kv=np.asarray(reference["kv"], np.float32),
        dtype=np.array(reference.get("dtype", "?")),
        steps=np.array(int(reference.get("steps", cfg.pack_steps))),
        threads=np.array(int(reference.get("threads", cfg.threads))),
        build_s=np.array(float(reference.get("build_s", 0.0))),
        timing=np.array(json.dumps(reference.get("timing", {}))),
        checks=np.array(json.dumps({k: round(v, 9) for k, v in checks.items()})),
        identity=np.array(json.dumps(identity)),
        content_id=np.array(content_id),
        baseline=np.array("self"),
    )
    log(
        "pack",
        f"wrote {out.name} ({_size_mb(out)} MB) so the notebook's accuracy section reuses it "
        f"(the pack's reference is this run, so the recorded check is cos "
        f"{checks['action_vs_pack_cos']:.8f} against itself - there is no other baseline on "
        f"the first run)",
    )
    return out


def _stage_and_run_reference(cfg, payload: dict) -> dict | None:
    """Bootstrap the torch reference for these inputs by running ``torch_ref_actions.py`` once.

    That script rebuilds the model from the clone and the checkpoint and replays the three graphs
    eagerly on the CPU (minutes on the first run, about 30 s once the checkpoint is in the page
    cache), reading ``--pack`` and writing ``--out``.  The staged pack carries the *inputs* only,
    because the reference is exactly what the run produces -- so the script skips its own
    self-checks there (``torch_ref_actions.py`` only compares when a reference is in the pack).

    Args:
        cfg: the run configuration (``art_dir``, ``workdir``, ``torch_ref_dtype``, ``pack_steps``,
            ``threads``).
        payload: the pack inputs assembled by :func:`step_pack`.

    Returns:
        ``{"action", "first_frame", "kv", "dtype", "steps", "threads", "build_s", "timing"}``, or
        ``None`` when the caller asked for no reference (``--torch-ref-dtype none``).
    """
    if cfg.torch_ref_dtype == "none":
        log("pack", "no torch reference requested (--torch-ref-dtype none): the pack carries the " "inputs only and the accuracy section will compute its own")
        return None
    if not TORCH_REF_SCRIPT.is_file():
        raise SystemExit(f"[packs] missing {TORCH_REF_SCRIPT}")
    pack_file = cfg.art_dir / "ov_runtime_pack.npz"
    out_file = cfg.art_dir / "torch_ref_pack.npz"
    np.savez_compressed(pack_file, **payload)  # inputs only: this run creates the reference
    cmd = [
        sys.executable,
        str(TORCH_REF_SCRIPT),
        "--workdir",
        str(cfg.workdir),
        "--pack",
        str(pack_file),
        "--out",
        str(out_file),
        "--dtype",
        cfg.torch_ref_dtype,
        "--steps",
        str(cfg.pack_steps),
        "--threads",
        str(cfg.threads),
    ]
    log(
        "pack",
        "running torch_ref_actions.py once on these inputs (PyTorch on the CPU - the slow " "part of the bootstrap; the result is cached for the notebook)",
    )
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("VIRTUAL_ENV", None)
    proc = subprocess.run(cmd, cwd=str(cfg.workdir), env=env, capture_output=True, text=True)
    for line in (proc.stdout or "").splitlines()[-4:]:
        log("torch", line)
    if proc.returncode != 0:
        tail = ((proc.stdout or "") + (proc.stderr or "")).splitlines()[-16:]
        raise SystemExit(f"[packs] torch_ref_actions.py failed ({proc.returncode})\n" + "\n".join(tail))
    with np.load(out_file) as z:
        return {
            "action": np.array(z["action"]),
            "first_frame": np.array(z["first_frame"]),
            "kv": np.array(z["kv"]),
            "dtype": str(z["dtype"]),
            "steps": int(z["steps"]),
            "threads": int(z["threads"]),
            "build_s": float(z["build_s"]),
            "timing": json.loads(str(z["timing"])),
        }


def step_pack(cfg) -> dict:
    """Assemble ``ov_runtime_pack.npz``: the one self-contained sample the runtime section runs.

    ``ctx_p`` is the 129-token stream the graphs take - 128 *zero* text tokens plus the projected
    proprio token, the same empty instruction the recorded pack carries, so the runtime section
    checks the three graphs and not the text path - ``ctx_pm`` its all-ones mask,
    ``t_sched``/``d_sched`` the flow-matching Euler schedule
    (``fwam_ov.build_schedule``, the same numbers the run uses), and ``action_ref`` the fp32 torch
    reference chunk for the very same inputs - plus the ``first_frame``/``kv00..15`` diagnostics
    ``torch_ref_actions.py`` self-checks against.  When the box has no ``torch_ref_pack.npz`` yet
    that reference is computed here (one ``torch_ref_actions.py`` run, cached for the notebook).
    """
    out = cfg.art_dir / "ov_runtime_pack.npz"
    needed = ["image", "lat_video", "lat_action", "ctx_p", "ctx_pm", "t_sched", "d_sched"]
    if not cfg.force and _have(out, needed + ["action_ref"]):
        log("pack", f"reusing {out.name} ({_size_mb(out)} MB)")
        return {"step": "pack", "status": "cache", "paths": [out], "seconds": 0.0}
    for name in ("eval_samples.npz", "eval_proprio.npz", "eval_latents.npz"):
        if not (cfg.art_dir / name).is_file():
            raise RuntimeError(f"missing artifacts/{name} - the pack is assembled from the " f"eval packs (run --steps samples,proprio,latents first)")

    sys.path.insert(0, str(NOTEBOOK))
    import fwam_ov  # noqa: PLC0415  (torch-free: the Euler schedule and its constants)

    samples = _load(cfg.art_dir / "eval_samples.npz")
    latents = _load(cfg.art_dir / "eval_latents.npz")
    image = np.transpose(samples["image_norm"][0], (2, 0, 1))[None].astype(np.float32)
    token = _proprio_token(samples["state_norm"][0], cfg.art_dir)[None, :, :]  # [1, 1, 4096]
    # The 128 text tokens are left at zero, an empty instruction: this pack checks the three graphs
    # against one torch chunk, and the recorded pack does the same, so the notebook's runtime section
    # reads identically whether the artifacts were generated here or came with the checkout.
    ctx_p = np.concatenate([np.zeros_like(token).repeat(CTX_SEQ, axis=1), token], axis=1).astype(np.float32)
    ctx_pm = np.ones((1, CTX_SEQ + 1), np.float32)  # 128 real tokens + the proprio token
    t_sched, d_sched = fwam_ov.build_schedule(cfg.pack_steps)
    payload = {
        "image": image,
        "lat_video": latents["lat_video"][0].astype(np.float32),
        "lat_action": latents["lat_action"][0].astype(np.float32),
        "ctx_p": ctx_p,
        "ctx_pm": ctx_pm,
        "t_sched": t_sched.astype(np.float32),
        "d_sched": d_sched.astype(np.float32),
    }
    # The reference has to belong to THIS sample: a cached one is only reused when it was computed
    # from these exact inputs, and otherwise it is recomputed from the pack inputs.
    content_id = _content_id(payload)
    reference = _cached_reference(cfg, content_id)
    bootstrapped = False
    if reference is not None:
        log("pack", f"reusing the torch reference at {cfg.art_dir / 'torch_ref_pack.npz'} " f"(computed from these inputs)")
    else:
        reference = _stage_and_run_reference(cfg, payload)
        bootstrapped = reference is not None
    if reference is not None:
        payload["action_ref"] = np.asarray(reference["action"], np.float32)
        payload["first_frame"] = np.asarray(reference["first_frame"], np.float32)
        kv = np.asarray(reference["kv"], np.float32)
        for index in range(kv.shape[0]):
            payload[f"kv{index:02d}"] = kv[index]
    np.savez_compressed(out, **payload)
    expect = {
        "image": ((1, 3, image.shape[-2], image.shape[-1]), "float32"),
        "lat_video": (VIDEO_LATENT_SHAPE, "float32"),
        "lat_action": (ACTION_LATENT_SHAPE, "float32"),
        "ctx_p": ((1, CTX_SEQ + 1, CTX_DIM), "float32"),
        "ctx_pm": ((1, CTX_SEQ + 1), "float32"),
        "t_sched": ((cfg.pack_steps,), "float32"),
        "d_sched": ((cfg.pack_steps,), "float32"),
    }
    if reference is not None:
        expect["action_ref"] = (ACTION_LATENT_SHAPE[1:], "float32")
        expect["first_frame"] = (FIRST_FRAME_SHAPE, "float32")
        expect["kv00"] = (VIDEO_KV_SHAPE, "float32")
    written = _verify(out, expect)
    if bootstrapped:
        _write_reference_cache(cfg, written, reference, content_id)
    log(
        "pack",
        f"wrote {out.name} ({_size_mb(out)} MB): ctx_p {ctx_p.shape} (128 zero text tokens + "
        f"the proprio token), " + ("action_ref present" if reference is not None else "no action_ref"),
    )
    return {"step": "pack", "status": "written", "paths": [out], "seconds": 0.0}


# --------------------------------------------------------------------------- main

STEP_FUNCS = {
    "samples": step_samples,
    "proprio": step_proprio,
    "contexts": step_contexts,
    "latents": step_latents,
    "pack": step_pack,
}


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--art-dir", default=None, help="where the packs are written (default <notebook>/artifacts)")
    parser.add_argument("--workdir", default=str(NOTEBOOK), help="the notebook folder (owns model/); default: this script's notebook")
    parser.add_argument("--steps", default="all", help=f"comma-separated subset of {','.join(STEPS)}, or all (default)")
    parser.add_argument("--force", action="store_true", help="rebuild packs that already exist")
    parser.add_argument("--compare-dir", default=None, help="optional reference directory: report per-array agreement after each write")
    parser.add_argument("--data-root", default=None, help=f"the released LIBERO tree (default <workdir>/model/FasterWAM/{RAW_DATA_REL})")
    parser.add_argument(
        "--no-download", action="store_true", help="never fetch the LIBERO episodes (they are downloaded on first use " "otherwise: 4.7 GB from Hugging Face)"
    )
    parser.add_argument(
        "--stats-rel", default="checkpoints/hustvl/FasterWAM/libero/dataset_stats.json", help="the checkpoint's dataset_stats.json, relative to the clone"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=T5_THREADS,
        help=f"torch/MKL threads (default {T5_THREADS}: the count that reproduces " f"the shipped contexts bit-for-bit)",
    )
    parser.add_argument("--noise-seed", dest="noise_seed", type=int, default=NOISE_SEED, help=f"seed of the recorded diffusion noise (default {NOISE_SEED})")
    parser.add_argument(
        "--torch-ref-dtype",
        dest="torch_ref_dtype",
        default="auto",
        choices=("auto", "float32", "bfloat16", "none"),
        help="dtype of the torch reference the pack bootstraps (none = do not run it)",
    )
    parser.add_argument(
        "--pack-steps", dest="pack_steps", type=int, default=PACK_STEPS, help=f"Euler steps of the runtime pack's schedule (default {PACK_STEPS})"
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested steps in dependency order and print one summary line."""
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    art_dir = Path(args.art_dir).resolve() if args.art_dir else workdir / "artifacts"
    steps = [s.strip().lower() for s in str(args.steps).split(",") if s.strip()]
    if steps == ["all"]:
        steps = list(STEPS)
    unknown = [s for s in steps if s not in STEP_FUNCS]
    if unknown:
        raise SystemExit(f"[packs] unknown steps {unknown}; pick from {list(STEPS)} or 'all'")
    steps = [s for s in STEPS if s in steps]  # dependency order, not user order
    art_dir.mkdir(parents=True, exist_ok=True)

    threads = set_threads(args.threads)
    cfg = argparse.Namespace(
        art_dir=art_dir,
        workdir=workdir,
        clone=workdir / "model" / "FasterWAM",
        ckpt=fwam_checkpoint_path(workdir),
        compare_dir=Path(args.compare_dir).resolve() if args.compare_dir else None,
        data_root=args.data_root,
        download=not args.no_download,
        stats_rel=args.stats_rel,
        threads=threads,
        noise_seed=args.noise_seed,
        pack_steps=args.pack_steps,
        torch_ref_dtype=args.torch_ref_dtype,
        force=bool(args.force),
    )
    log("packs", f"steps {steps} -> {art_dir} ({'force' if cfg.force else 'reuse what exists'})")
    t0 = time.perf_counter()
    summary = []
    for name in steps:
        step_t0 = time.perf_counter()
        result = STEP_FUNCS[name](cfg)
        result.setdefault("seconds", round(time.perf_counter() - step_t0, 1))
        for path in result["paths"]:
            _compare(Path(path), cfg.compare_dir)
        summary.append({"step": name, "status": result.get("status", "written"), "seconds": result.get("seconds", 0.0)})
    total = round(time.perf_counter() - t0, 1)
    log("packs", f"done in {total} s: " + ", ".join(f"{s['step']}={s['status']}" for s in summary))
    print("SUMMARY " + json.dumps({"art_dir": str(art_dir), "steps": summary, "seconds": total}), flush=True)
    return 0


def fwam_checkpoint_path(workdir: Path) -> Path:
    """Return the released LIBERO checkpoint the notebook uses (the clone's HF cache layout)."""
    return Path(workdir) / "model" / "FasterWAM" / "checkpoints" / "hustvl" / "FasterWAM" / "libero" / "step_021700.pt"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
