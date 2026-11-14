"""Torch-free FasterWAM to OpenVINO runtime and IR cache, used by ``fasterWAM.ipynb``.

One action chunk is three OpenVINO graphs plus a ten step loop::

    vae     image[1,3,1,224,448] -------------------------------------------------> latent[1,48,1,14,28]
    video   latents[1,48,3,14,28] + timestep[1] + context[1,129,4096] + mask[1,129] -> 16 fused K/V
    action  latents_action[1,32,7] + t[1] + context + 16 K/V ----------------------> pred[1,32,7]

The video graph taps the K/V of every fourth of its 30 blocks (``COND_LAYERS``), so its 16 outputs
*are* the conditioning the action graph needs: one video forward pass per chunk, not a video
diffusion loop.  The action graph is then applied ``steps`` times (flow matching Euler:
``x += pred * delta``).  The 16 K/V tensors and the context are constant across those steps, so
they are cast - and, when the device allows it, uploaded - to the action graph's input precision once
per chunk (re-casting ~57 MB per step used to cost about 9 ms per step).

Two runtime levers are implemented, on by default on a GPU and switchable through
``load_units(..., kvdev=..., overlap=...)``:

  * ``kvdev`` - the 16 video K/V plus the context live in DEVICE memory as ``RemoteContext`` tensors,
    so the 10 Euler steps never re-read them from the host.  It needs an action graph with **f16**
    inputs, which only the INT8 set has (its video and action units are traced in half precision).
    The fp16 set is an fp32 trace whose weights are compressed when saved, so its graph inputs are
    f32 and this lever reports itself unavailable with one printed line - by design, not an error.
  * ``overlap`` - the video prefill is issued with ``start_async`` on its own InferRequest, and the
    context cast / upload runs while it is in flight.  It is also what makes the fp16 set's action
    loop fast: the action steps are then fed the prefill request's own output tensors, and measured on
    an integrated GPU that takes the steps from ~22 to ~17.5 ms each (-45 ms per chunk), the same as
    copying the f32 K/V into device memory would - so the fp16 set needs no kvdev.

Both change only the memory domain and the order of host-side work, never the graph or the bytes, so
the actions are bit-identical.  The action-context tower (``actx``) is **not** implemented here: it
needs the ``action_b1`` split graphs, which only the INT8 export writes, and it is worth a further
~5 ms, so it is left out to keep this module small.

The kernel that uses this module never imports torch.  The export runs in a subprocess, so do the
recorded evaluation packs the notebook starts from (:func:`ensure_eval_artifacts` ->
``utils/fwam_export/make_eval_packs.py``), and so does the accuracy section's reference chunk
(:func:`ensure_torch_ref` -> ``torch_ref_actions.py``), which rebuilds the model in PyTorch on the
CPU and replays the same three graphs eagerly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# The video graph taps the K/V of these transformer blocks (every fourth of 30) -> 8 layers.
COND_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28)
UNITS = ("vae", "video", "action")
PRECISIONS = ("fp16", "int8")

# --------------------------------------------------------------- export switches
# Graph-shaping export switches: a different flag set is a different graph, hence a different cache
# entry (see ir_root_for).  The notebook always exports the fully promoted graph below - there is no
# switch panel - but every optimization keeps its own exporter flag (``--rope3``, ``--rope2``,
# ``--tmod2``, ``--modg``, ``--no-vae-fold``), so the code that applies it can be read, re-measured
# and rolled back from outside the notebook.  ``dtype`` is always fp16; the precision dropdown only
# decides the INT8 weight-compression switches, and for int8 adds ``--trace-half`` to the video and
# action units (see _export_cmd).
DEFAULT_EXPORT_FLAGS = {
    "dtype": "fp16",
    "rope2": 1,  # pair-swap-free RoPE: 3 kernels per application instead of 9 (bit-identical)
    "rope3": 1,  # half-major RoPE basis -> one fused kernel; takes precedence over rope2
    "tmod2": 1,  # video t_mod from its 2 distinct timesteps (M=2) instead of all 294
    "modg": 1,  # t_mod as one row per latent frame; needs tmod2; deletes 28 per-block adds
    "vae_fold": True,  # fold the VAE encoder's 22 dead Conv3D depth taps (233 -> 81 GMAC)
}
EXPORT_FLAGS = dict(DEFAULT_EXPORT_FLAGS)

# The recorded evaluation inputs and reference actions (LIBERO episodes) live in the notebook's own
# ``artifacts/`` folder.  They are *generated on first use* rather than committed with the notebook
# (tens of megabytes of them), by :func:`ensure_eval_artifacts`; the folder is git-ignored.  The
# PyTorch source is not taken from there - the notebook clones it (see :func:`clone_source`).
# The keys each pack must carry: the cache check below trusts a file only when it parses and holds
# them, so a truncated or half-written pack is regenerated instead of failing three cells later.
ARTIFACT_KEYS = {
    "eval_samples.npz": ("image", "state_norm", "gt_action_norm"),
    "eval_samples.json": (),  # only required to be readable JSON with a non-empty `samples`
    "eval_contexts.npz": ("context", "mask"),
    "eval_latents.npz": ("lat_video", "lat_action"),
    "eval_proprio.npz": ("weight", "bias"),
    "ov_runtime_pack.npz": ("image", "lat_video", "lat_action", "ctx_p", "ctx_pm", "t_sched", "d_sched", "action_ref"),
}

ARTIFACT_FILES = (
    "eval_contexts.npz",
    "eval_latents.npz",
    "eval_proprio.npz",
    "eval_samples.json",
    "eval_samples.npz",
    "ov_runtime_pack.npz",
)


# --------------------------------------------------------------------------- paths


def artifacts_dir(workdir: str | Path) -> Path:
    """Return the notebook's ``artifacts/`` folder and check that every recorded file is in it.

    Args:
        workdir: the notebook's directory.

    Returns:
        ``<workdir>/artifacts``.

    Raises:
        FileNotFoundError: when the folder or any of :data:`ARTIFACT_FILES` is missing.
    """
    art = Path(workdir) / "artifacts"
    missing = [name for name in ARTIFACT_FILES if not (art / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{art} is missing {', '.join(missing)} - run fwam_ov.ensure_eval_artifacts(workdir) " f"to generate them")
    return art


def _artifacts_broken(art: Path) -> list[str]:
    """Return the packs of :data:`ARTIFACT_FILES` that are missing, unreadable or lack a key.

    Presence and a non-zero size are not enough: an interrupted copy or a full disk leaves a file
    that opens but has no arrays, and the notebook would then fail inside a later cell instead of
    here.  Only the directories' entries are read for ``.npz`` (``np.load`` does not decompress the
    arrays), so this stays cheap on a cache hit.
    """
    broken: list[str] = []
    for name, keys in ARTIFACT_KEYS.items():
        path = art / name
        if not path.is_file() or path.stat().st_size == 0:
            broken.append(name)
            continue
        try:
            if path.suffix == ".json":
                if not json.loads(path.read_text()).get("samples"):
                    broken.append(name)
            else:
                with np.load(path, allow_pickle=False) as pack:
                    if any(key not in pack.files for key in keys):
                        broken.append(name)
        except (OSError, ValueError):
            broken.append(name)
    return broken


# The generator's steps, in dependency order, and the artifacts each one writes.  ``samples`` is the
# only step that needs external data (the LIBERO episodes themselves).
EVAL_ARTIFACT_STEPS = ("samples", "proprio", "contexts", "latents", "pack")
EVAL_STEP_OUTPUTS = {
    "samples": ("eval_samples.npz", "eval_samples.json"),
    "proprio": ("eval_proprio.npz",),
    "contexts": ("eval_contexts.npz",),
    "latents": ("eval_latents.npz",),
    "pack": ("ov_runtime_pack.npz",),
}


def ensure_eval_artifacts(workdir: str | Path, steps: str | list[str] | tuple[str, ...] | None = None, force: bool = False, log: bool = True) -> dict:
    """Return the notebook's evaluation packs, generating them into ``artifacts/`` when absent.

    The packs are generated on first use instead of being committed (tens of MB next to a 1 MB notebook):
    ``utils/fwam_export/make_eval_packs.py`` rebuilds every array from the released checkpoint, the
    umT5 weights and the LIBERO evaluation episodes, in the same five idempotent steps.  Like the
    export and the torch reference it runs as a subprocess, so this kernel never imports torch.

    Four steps are self-contained once ``model/`` exists - ``latents`` is a seeded draw, ``proprio``
    reads the checkpoint, ``contexts`` encodes the 32 instructions with umT5, and ``pack`` assembles
    the one self-contained sample the runtime section runs plus its PyTorch reference (one CPU run,
    cached in ``torch_ref_pack.npz``).  ``samples`` is the exception: the observations are real
    episodes, so the generator fetches the LIBERO LeRobot tree from the Hugging Face dataset
    ``yuanty/LIBERO-fastwam`` on first use (4.7 GB: the unpacked ``lerobot_v30/`` tree, not the
    repository's tarballs).  That is the only download, it happens once, and it is skipped when the
    episodes are already there.

    Args:
        workdir: the notebook's directory (owns ``model/`` and ``artifacts/``).
        steps: which steps to run; all five by default (cell 9 reads ``eval_samples``, so the
            notebook always asks for everything - a subset is only useful for hand work).
        force: regenerate even when every pack is already there.
        log: forward the subprocess output as it arrives.

    Returns:
        ``{"art_dir", "paths", "source", "seconds"}``: the ``artifacts`` folder, its six names
        mapped to their paths (in :data:`ARTIFACT_FILES` order), ``"cache"`` when nothing had to be
        done or ``"generated"`` after a run, and the seconds that run took.  A pack that parses but
        lacks a key of :data:`ARTIFACT_KEYS` counts as missing and is regenerated.
    """
    workdir = Path(workdir).resolve()
    art = workdir / "artifacts"
    if steps is None:
        wanted = list(EVAL_ARTIFACT_STEPS)
    else:
        wanted = [part.strip().lower() for part in (steps.split(",") if isinstance(steps, str) else steps) if part.strip()]
        if wanted == ["all"]:
            wanted = list(EVAL_ARTIFACT_STEPS)
        unknown = [name for name in wanted if name not in EVAL_ARTIFACT_STEPS]
        if unknown:
            raise ValueError(f"unknown steps {unknown}; pick from {list(EVAL_ARTIFACT_STEPS)} or 'all'")

    paths = {name: art / name for name in ARTIFACT_FILES}
    broken = _artifacts_broken(art)
    if not force and not broken:
        if log:
            size = sum(path.stat().st_size for path in paths.values()) / 1e6
            print(f"[packs] reusing the evaluation artifacts in {art.relative_to(workdir)} " f"({size:.0f} MB, {len(ARTIFACT_FILES)} files)")
        return {"art_dir": art, "paths": paths, "source": "cache", "seconds": 0.0}

    if not MAKE_PACKS_SCRIPT.is_file():
        raise FileNotFoundError(f"missing {MAKE_PACKS_SCRIPT}")
    if log:
        print(
            f"[packs] generating the evaluation artifacts in {art} ({len(broken)} of " f"{len(ARTIFACT_FILES)} missing or incomplete: {', '.join(broken)})",
            flush=True,
        )
        print(
            "[packs] four of the five steps only need the checkpoint and umT5; the observations "
            "come from the LIBERO episodes (see this cell's output if they are absent)",
            flush=True,
        )

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("VIRTUAL_ENV", None)  # keep the subprocess in this notebook's environment
    cmd = [sys.executable, str(MAKE_PACKS_SCRIPT), "--workdir", str(workdir), "--art-dir", str(art), "--steps", ",".join(wanted)]
    if force:
        cmd.append("--force")
    t0 = time.perf_counter()
    tail: list[str] = []
    code = _run_streaming(cmd, workdir, env, keep_tail=80, tail=tail)
    seconds = round(time.perf_counter() - t0, 1)
    if code != 0:
        hint = ""
        if code < 0:  # SIGKILL: almost always the OOM killer on a small host
            hint = (
                f"\n[signal {-code}: the generator was killed, most likely for memory - the umT5 encode "
                f"peaks near 20 GB and the fp32 PyTorch reference near 34 GB. Run it on a host with "
                f"more RAM, or one step at a time: --steps proprio,latents,contexts, then --steps pack]"
            )
        raise RuntimeError(f"generating the evaluation artifacts failed (exit {code}):{hint}\n" + "\n".join(tail[-20:]))
    missing = [name for step in wanted for name in EVAL_STEP_OUTPUTS[step] if not paths[name].is_file()]
    if missing:
        raise RuntimeError(f"the generator reported success, but artifacts/ is still missing " f"{', '.join(missing)}")
    if log:
        print(f"[packs] generated in {seconds} s")
    return {"art_dir": art, "paths": paths, "source": "generated", "seconds": seconds}


# --------------------------------------------------------------------------- source and checkpoint

REPO_URL = "https://github.com/hustvl/FasterWAM.git"
# The source commit the bundled exporter was written and validated against.  The exporter
# monkeypatches the repo's modules, so a newer upstream commit may no longer match it.
REPO_COMMIT = "83667817df0d4f823f39d90700e61ea2f432ac45"
CKPT_HF_REPO = "hustvl/FasterWAM"
CKPT_HF_REVISION = "6bf9471ced6919a15ab8fded89f7772f5060c44b"  # the HF commit the IR was exported from
CKPT_HF_ALLOW = ("libero/*",)  # only the LIBERO checkpoint; the robotwin one is not exported here
CKPT_NAME = "step_021700.pt"

# The OpenVINO exporter bundled with this notebook (it monkeypatches the cloned repo, it is not
# part of the upstream repository).
EXPORT_DIR = Path(__file__).resolve().parent / "utils" / "fwam_export"
EXPORT_SCRIPT = EXPORT_DIR / "export_openvino.py"
MAKE_REF_SCRIPT = EXPORT_DIR / "make_ref.py"
REF_TASK = "libero_fasterwam_2cam224_1e-4"  # the LIBERO task the notebook exports
EXPORT_ARTIFACTS = EXPORT_DIR / "artifacts"  # exporter's own cache: input dump + derived refs

# The PyTorch-on-the-CPU reference the accuracy section compares against.  Like the exporter it runs
# as a subprocess - the notebook's kernel never imports torch.
TORCH_REF_SCRIPT = Path(__file__).resolve().parent / "torch_ref_actions.py"
TORCH_REF_NAME = "torch_ref_pack.npz"

# The generator of the recorded evaluation packs, ``artifacts/*.npz`` (:func:`ensure_eval_artifacts`).
MAKE_PACKS_SCRIPT = EXPORT_DIR / "make_eval_packs.py"


def model_dir(workdir: str | Path) -> Path:
    """Return the notebook's ``model/`` folder (source clone, checkpoint and exported IR sets)."""
    return Path(workdir) / "model"


def source_dir(workdir: str | Path) -> Path:
    """Return the directory of the notebook's own FasterWAM source clone (``<workdir>/model/FasterWAM``)."""
    return model_dir(workdir) / "FasterWAM"


def checkpoint_path(workdir: str | Path) -> Path:
    """Return the checkpoint path the bundled exporter expects under the clone."""
    return source_dir(workdir) / "checkpoints" / CKPT_HF_REPO / "libero" / CKPT_NAME


def _git(clone: Path, *args: str) -> subprocess.CompletedProcess:
    """Run one git command inside ``clone`` and capture its output."""
    return subprocess.run(["git", "-C", str(clone), *args], capture_output=True, text=True)


def clone_source(workdir: str | Path, url: str = REPO_URL, force: bool = False, log: bool = True, commit: str | None = REPO_COMMIT) -> Path:
    """Fetch the FasterWAM PyTorch repository, at a pinned commit, into ``<workdir>/model/FasterWAM``.

    Only that one commit is fetched (``git fetch --depth 1 origin <sha>``), so the source matches what
    the bundled exporter was validated against however far upstream has moved.  An existing clone is
    reused as it is; when its HEAD is not the pinned commit one warning line is printed.

    Args:
        workdir: the notebook's directory (owns ``model/``).
        url: git URL of the public FasterWAM repository.
        force: delete an existing clone and fetch again.
        log: print one line per decision.
        commit: the commit to check out (default :data:`REPO_COMMIT`); ``None`` takes the default
            branch's tip instead.

    Returns:
        The clone path.
    """
    clone = source_dir(workdir)
    clone.mkdir(parents=True, exist_ok=True)
    if (clone / ".git").is_dir():
        if not force:
            head = _git(clone, "rev-parse", "HEAD").stdout.strip()
            if log:
                print(f"[src] reusing the FasterWAM clone at {clone} (commit {head[:7] or '?'})")
                if commit and head and head != commit:
                    print(f"[src] warning: the clone is not at the pinned commit {commit[:7]} - " f"the bundled exporter was validated against that one")
            return clone
        shutil.rmtree(clone)
        clone.mkdir(parents=True)
    if any(clone.iterdir()):
        raise RuntimeError(f"{clone} is not empty and not a git clone - clear it first")
    if log:
        print(f"[src] fetching {url} @ {commit[:7] if commit else 'HEAD'} into {clone} ...")
    if commit:
        steps = (["init", "-q"], ["remote", "add", "origin", url], ["fetch", "-q", "--depth", "1", "origin", commit], ["checkout", "-q", "FETCH_HEAD"])
        for args in steps:
            proc = _git(clone, *args)
            if proc.returncode != 0:
                raise RuntimeError(f"git {' '.join(args)} failed:\n{proc.stderr.strip()[-1500:]}")
    else:
        proc = subprocess.run(["git", "clone", "--depth", "1", url, str(clone)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{proc.stderr.strip()[-1500:]}")
    if not (clone / "src" / "fasterwam").is_dir():
        raise RuntimeError(f"clone at {clone} has no src/fasterwam - unexpected repository layout")
    if log:
        print(f"[src] cloned into {clone}")
    return clone


def download_checkpoint(workdir: str | Path, force: bool = False, log: bool = True, revision: str | None = CKPT_HF_REVISION) -> Path:
    """Download the released FasterWAM LIBERO checkpoint from Hugging Face into the clone.

    The bundled exporter loads its weights solely from this file
    (``skip_dit_load_from_pretrain=True``), so no Wan2.2 components are needed.

    Args:
        workdir: the notebook's directory (owns ``model/``).
        force: re-download even when the checkpoint file is already there.
        log: print one line per decision.
        revision: the Hugging Face commit to download (default :data:`CKPT_HF_REVISION`, the one the
            notebook was validated with); ``None`` takes the repository's current ``main``.

    Returns:
        The checkpoint ``.pt`` path.
    """
    ckpt = checkpoint_path(workdir)
    if ckpt.is_file() and not force:
        if log:
            print(f"[ckpt] reusing {ckpt} ({ckpt.stat().st_size / 1e9:.2f} GB)")
        return ckpt
    clone = source_dir(workdir)
    if not (clone / "src" / "fasterwam").is_dir():
        raise RuntimeError("no FasterWAM clone - run clone_source() first")
    from huggingface_hub import snapshot_download

    if log:
        print(f"[ckpt] downloading {CKPT_HF_REPO}@{(revision or 'main')[:7]} ({', '.join(CKPT_HF_ALLOW)}) " f"into {clone / 'checkpoints'} ...")
    snapshot_download(repo_id=CKPT_HF_REPO, revision=revision, allow_patterns=list(CKPT_HF_ALLOW), local_dir=str(clone / "checkpoints" / CKPT_HF_REPO))
    if not ckpt.is_file():
        raise RuntimeError(f"download finished but {ckpt} is missing - unexpected HF repository layout")
    if log:
        print(f"[ckpt] {ckpt} ({ckpt.stat().st_size / 1e9:.2f} GB)")
    return ckpt


def ir_paths(ir_root: Path, precision: str = "fp16") -> dict[str, Path]:
    """Return the three IR xml files of one IR set, tolerating both known directory layouts.

    The current exporter writes the IR FLAT inside the unit directory
    (``<ir_root>/<unit>/<unit>_fp16.{xml,bin}``, with ``action_b1/`` beside it); sets exported
    before 12 Nov carry an extra ``<unit>/ir/`` level.  The flat layout is looked for first, the
    legacy one second, so both keep working.

    Args:
        ir_root: IR set root, for example ``<notebook>/model/ir/int8``.
        precision: ``fp16`` or ``int8``; the files are named ``<unit>_<precision>.xml``.  A set
            that still carries the exporter's ``<unit>_fp16.xml`` names is found too, and
            :func:`rename_ir_files` gives it the precision's own names.

    Returns:
        ``{"vae": ..., "video": ..., "action": ...}``.
    """
    ir_root = Path(ir_root)
    out = {}
    for unit in UNITS:
        found = None
        for name in (f"{unit}_{precision}.xml", f"{unit}_fp16.xml"):
            for sub in (unit, f"{unit}/ir"):  # flat first, legacy `<unit>/ir/` as the fallback
                if (ir_root / sub / name).is_file():
                    found = found or ir_root / sub / name
        out[unit] = found or ir_root / unit / f"{unit}_{precision}.xml"
    return out


def rename_ir_files(ir_root: Path, precision: str) -> None:
    """Rename an exported set's ``<unit>_fp16.{xml,bin}`` to ``<unit>_<precision>.{xml,bin}``.

    The exporter always writes ``*_fp16.*``; an INT8 set should not carry that name.  The ``.xml``
    refers to its weights only by file position, so renaming the pair is safe.  Does nothing for
    fp16, and skips a unit whose new name already exists.  Both the current FLAT layout and the
    pre-12-Nov ``<unit>/ir/`` one are checked.
    """
    if precision == "fp16":
        return
    for unit in UNITS:
        for sub in (unit, f"{unit}/ir"):
            old = Path(ir_root) / sub / f"{unit}_fp16.xml"
            new = old.with_name(f"{unit}_{precision}.xml")
            if old.is_file() and not new.exists():
                old.rename(new)
                old.with_suffix(".bin").rename(new.with_suffix(".bin"))


def set_complete(paths: dict[str, Path]) -> bool:
    """Return True when every unit of an IR set is present on disk."""
    return all(p.is_file() and p.with_suffix(".bin").is_file() for p in paths.values())


def set_fingerprint(paths: dict[str, Path]) -> dict[str, dict]:
    """Return a small, honest description of an IR set: xml hash plus bin size per unit.

    Args:
        paths: output of :func:`ir_paths`.

    Returns:
        ``{unit: {"xml_md5": ..., "xml_bytes": ..., "bin_bytes": ...}}``.  The ``.xml`` file
        carries the graph structure, so its hash separates an INT8 set (compressed
        ``FullyConnectedCompressed`` weights) from the fp16 one even though the file names match.
    """
    out = {}
    for unit, xml in paths.items():
        if not xml.is_file():
            continue
        digest = hashlib.md5(xml.read_bytes()).hexdigest()
        bin_ = xml.with_suffix(".bin")
        out[unit] = {
            "xml_md5": digest,
            "xml_bytes": xml.stat().st_size,
            "bin_bytes": bin_.stat().st_size if bin_.is_file() else 0,
        }
    return out


def _flag_hash(precision: str) -> str:
    """Return a stable hash of the precision plus the graph-shaping export flags."""
    payload = {"precision": precision, **EXPORT_FLAGS}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _at_default_flags() -> bool:
    """Return True when ``EXPORT_FLAGS`` still describes the fully promoted (default) graph."""
    return all(EXPORT_FLAGS.get(k) == v for k, v in DEFAULT_EXPORT_FLAGS.items())


def _graph_flag_hash() -> str:
    """Return a stable hash of the graph-shaping export flags alone (no precision).

    The exporter's derived torch references depend on the graph flags but not on the weight
    precision, so they are keyed on this and shared by the fp16 and the INT8 export.
    """
    return hashlib.sha1(json.dumps(EXPORT_FLAGS, sort_keys=True).encode()).hexdigest()[:12]


def _ref_sidecar(unit: str) -> Path:
    """Return the file that records which graph flags a unit's derived torch reference was made with."""
    return EXPORT_ARTIFACTS / f"ov_{unit}_ref.flags.json"


def _can_reuse_ref(unit: str) -> bool:
    """Return True when the exporter may reuse its derived torch reference for ``unit``.

    Only a reference derived with the current graph flags is reusable.  A reference without a
    sidecar predates the sidecar and was made by the default export, so it is reused only while the
    flags are still the defaults.
    """
    if not (EXPORT_ARTIFACTS / f"ov_{unit}_ref.pt").is_file():
        return False
    sidecar = _ref_sidecar(unit)
    if not sidecar.is_file():
        return _at_default_flags()
    try:
        return json.loads(sidecar.read_text()).get("graph_flag_hash") == _graph_flag_hash()
    except (json.JSONDecodeError, OSError, AttributeError):
        return False


def _write_ref_sidecar(unit: str) -> None:
    """Record the graph flags of the reference the exporter just derived (or reused) for ``unit``."""
    if (EXPORT_ARTIFACTS / f"ov_{unit}_ref.pt").is_file():
        _ref_sidecar(unit).write_text(json.dumps({"graph_flag_hash": _graph_flag_hash(), "flags": EXPORT_FLAGS}, indent=2))


def export_deltas() -> dict[str, str]:
    """Return, per switch, whether the current value is the promoted default or a deviation."""
    return {key: "default" if EXPORT_FLAGS.get(key) == default else f"{EXPORT_FLAGS.get(key)!r}" for key, default in DEFAULT_EXPORT_FLAGS.items()}


def ir_root_for(workdir: str | Path, precision: str) -> Path:
    """Return the directory the current flag set exports into.

    The fully promoted default keeps the historical ``<model>/ir/<precision>``, so a set exported by
    an earlier run of this notebook is still found and reused.  A non-default flag set (``EXPORT_FLAGS``
    edited by hand; ``modg`` needs ``tmod2``) gets its own ``<model>/ir/<precision>__<flag hash>``
    directory, so reproducing a rollback set never overwrites the default one.

    Args:
        workdir: the notebook's directory, which owns the ``model/`` folder.
        precision: ``fp16`` or ``int8``.

    Returns:
        The IR set root for these flags.
    """
    root = model_dir(workdir) / "ir"
    if _at_default_flags():
        return root / precision
    return root / f"{precision}__{_flag_hash(precision)}"


def manifest_path_for(workdir: str | Path, precision: str) -> Path:
    """Return the manifest file that describes this flag set's export."""
    mdir = model_dir(workdir)
    if _at_default_flags():
        return mdir / f"ir_manifest_{precision}.json"
    return mdir / f"ir_manifest_{precision}__{_flag_hash(precision)}.json"


def export_plan(precision: str, workdir: str | Path | None = None) -> dict[str, list[str]]:
    """Return the exporter argv per unit for the current flags; nothing is executed.

    Lets the notebook show the exact command its fixed flag set resolves to, including the switches
    the precision dropdown implies (``--video-int8`` / ``--action-int8`` / ``--trace-half``).

    Args:
        precision: ``fp16`` or ``int8``.
        workdir: the notebook's directory (only used to render the output path).

    Returns:
        ``{unit: argv}``.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    out_dir = ir_root_for(workdir, precision) if workdir is not None else Path("<model>/ir") / precision
    ref = EXPORT_ARTIFACTS / f"ref_inputs_{REF_TASK}.pt"
    return {unit: _export_cmd(unit, precision, out_dir, ref) for unit in UNITS}


# --------------------------------------------------------------------------- export


def _export_cmd(unit: str, precision: str, out_dir: Path, ref_dump: Path) -> list[str]:
    """Build the subprocess command that exports one unit of one precision set.

    Args:
        unit: one of ``vae``, ``video``, ``action``.
        precision: ``fp16`` or ``int8``.
        out_dir: IR set root to write into (the exporter writes ``<out_dir>/<unit>/<unit>_fp16.xml``).
        ref_dump: synthetic input dump from :func:`_ensure_ref_dump` - the exporter re-derives
            its torch references from these inputs and checks every wrapper against them.

    Returns:
        The argv to run.  ``sys.executable`` pins the interpreter to the one running the kernel
        (the notebook virtual environment with the CPU Torch); ``--reuse-ref`` is added only when
        this exporter folder holds a reference for the unit derived with the same graph flags
        (see :func:`_can_reuse_ref`).
    """
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--unit",
        unit,
        "--dtype",
        EXPORT_FLAGS["dtype"],
        "--out-dir",
        str(out_dir),
        "--rope2",
        str(EXPORT_FLAGS["rope2"]),
        "--rope3",
        str(EXPORT_FLAGS["rope3"]),
        "--tmod2",
        str(EXPORT_FLAGS["tmod2"]),
        "--modg",
        str(EXPORT_FLAGS["modg"]),
    ]
    cmd += ["--vae-fold"] if EXPORT_FLAGS["vae_fold"] else ["--no-vae-fold"]
    # The INT8 weight-compression switches default to ON in the exporter for `video` and `action`;
    # ask for them explicitly in both directions so the command says what it means.
    want_int8 = precision == "int8"
    if unit == "video":
        cmd += ["--video-int8" if want_int8 else "--no-video-int8"]
    if unit == "action":
        cmd += ["--action-int8" if want_int8 else "--no-action-int8"]
    # INT8 compression needs the weights to be plain Constants feeding each MatMul, which only a
    # half-precision trace produces (an fp32 trace compressed at save time leaves a Convert in
    # between, and the exporter then finds no compressible MatMul). It also halves the ONNX scratch.
    if want_int8 and unit in ("video", "action"):
        cmd += ["--trace-half"]
    # The exporter checks every traced wrapper against a torch reference that it derives from the
    # inputs in this dump.  The reference depends on the graph flags but not on the precision, so
    # one derived with the same graph flags (any precision) is reused.
    cmd += ["--ref", str(ref_dump)]
    if _can_reuse_ref(unit):
        cmd += ["--reuse-ref"]
    return cmd


def make_ref_dump(workdir: str | Path, log: bool = True) -> Path:
    """Generate the synthetic input dump that the exporter derives its torch references from.

    ``make_ref.py`` reads the task config of the clone and builds seed-reproducible inputs, so it
    needs neither the checkpoint nor a model build - only the shapes.  The dump is written into
    ``utils/fwam_export/artifacts`` and reused.

    Args:
        workdir: the notebook's directory, which owns the ``model/`` clone.
        log: print the subprocess output.

    Returns:
        The ``.pt`` dump path.
    """
    clone = source_dir(workdir)
    if not (clone / "src" / "fasterwam").is_dir():
        raise RuntimeError(f"no FasterWAM source clone at {clone} - run clone_source() first")
    EXPORT_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dump = EXPORT_ARTIFACTS / f"ref_inputs_{REF_TASK}.pt"
    if dump.is_file():
        if log:
            print(f"[ref] reusing the input dump {dump}")
        return dump
    env = dict(os.environ, REPO=str(clone), PYTHONPATH=str(clone / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("VIRTUAL_ENV", None)
    cmd = [sys.executable, str(MAKE_REF_SCRIPT), "--task", REF_TASK, "--out", str(dump)]
    if log:
        print(f"[ref] building the input dump for task {REF_TASK} ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(clone), env=env)
    if proc.returncode != 0 or not dump.is_file():
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
        raise RuntimeError(f"input dump failed:\n{tail}")
    if log:
        print(f"[ref] {dump}")
    return dump


ONNX_SCRATCH = "onnx"  # the exporter's Torch->ONNX scratch, written beside each unit's IR


def _drop_onnx_scratch(ir_root: str | Path, units: tuple[str, ...] = UNITS, log: bool = True) -> float:
    """Delete the Torch->ONNX scratch an exporter run leaves beside the IR. Returns the GB freed.

    The exporter traces each unit through ONNX (``<ir_root>/<unit>/onnx``: the graph plus external
    data - 18 GB of it for the video unit, about 21 GB for a complete fp16 set) and never reads it
    again once the IR has been written, so it is pure scratch. Dropping it per unit keeps the peak
    near one unit's scratch instead of the whole set's.
    """
    freed = 0
    for unit in units:
        scratch = Path(ir_root) / unit / ONNX_SCRATCH
        if not scratch.is_dir():
            continue
        size = sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file())
        shutil.rmtree(scratch, ignore_errors=True)
        freed += size
        if log and size:
            print(f"[ir] {unit}: removed the ONNX scratch ({size / 1e9:.2f} GB) - the IR is self-contained")
    return round(freed / 1e9, 2)


def ensure_ir(
    precision: str,
    workdir: str | Path,
    force: bool = False,
    log: bool = True,
    keep_onnx: bool = False,
) -> dict:
    """Return a complete IR set for one precision, exported into ``<workdir>/model/ir/<precision>``.

    The set is built by the exporter bundled in ``utils/fwam_export`` against the FasterWAM clone
    in ``<workdir>/model/FasterWAM`` and the checkpoint in that clone's ``checkpoints/`` - both of
    which the notebook fetches itself; no IR from anywhere else is ever read.  Non-default
    ``EXPORT_FLAGS`` export into ``<workdir>/model/ir/<precision>__<flag hash>`` instead (see
    :func:`ir_root_for`).  Each unit's traced wrapper is checked against a torch reference derived
    from a synthetic, seed-fixed input dump (:func:`make_ref_dump`); derived references are cached
    in ``utils/fwam_export/artifacts`` with a sidecar recording their graph flags, and reused across
    precisions only while those flags match.

    Args:
        precision: ``fp16`` or ``int8``.
        workdir: the notebook's directory, which owns the ``model/`` folder.
        force: re-export even when the IR set is already complete on disk.
        log: print one line per decision.
        keep_onnx: keep the ``<ir_root>/<unit>/onnx`` scratch the exporter writes (about 21 GB for
            a complete set) instead of deleting each unit's as soon as its IR is on disk.
            ``FWAM_KEEP_ONNX=1`` in the environment has the same effect.

    Returns:
        ``{"precision", "paths", "fingerprint", "source", "seconds", "cached"}`` where ``source``
        is ``"cache"`` or ``"export"``.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    workdir = Path(workdir).resolve()
    ir_root = ir_root_for(workdir, precision)
    manifest_path = manifest_path_for(workdir, precision)
    keep_onnx = keep_onnx or os.environ.get("FWAM_KEEP_ONNX", "").strip().lower() in ("1", "true", "yes")
    fingerprint_flags = _flag_hash(precision)
    if log:
        deviating = [k for k, v in export_deltas().items() if v != "default"]
        print(
            f"[ir] {precision}: flags {'all promoted defaults' if not deviating else ', '.join(f'{k}={EXPORT_FLAGS[k]}' for k in deviating)}" f" -> {ir_root}"
        )

    # A manifest is only trusted while it describes this notebook's own export directory.
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            loaded = {}
        if loaded.get("ir_root") and Path(loaded["ir_root"]).resolve() == ir_root:
            manifest = loaded

    rename_ir_files(ir_root, precision)
    paths = ir_paths(ir_root, precision)
    # The manifest is written only after every unit exported, so a set without one (or whose files
    # no longer match it) is an interrupted or foreign export: re-export rather than compile it.
    reuse = not force and set_complete(paths) and manifest.get("flag_hash") == fingerprint_flags and set_fingerprint(paths) == manifest.get("fingerprint")
    if log and not force and set_complete(paths) and not reuse:
        print(f"[ir] {precision}: the set on disk has no matching manifest (interrupted export?) - exporting again")

    if reuse:
        if log:
            print(f"[ir] {precision}: reusing the exported set at {ir_root}")
        source, seconds = "cache", 0.0
        if not keep_onnx:
            _drop_onnx_scratch(ir_root, UNITS, log=log)  # scratch left by an interrupted run
    else:
        clone = source_dir(workdir)
        if not (clone / "src" / "fasterwam").is_dir():
            raise RuntimeError(f"no FasterWAM source clone at {clone} - run clone_source() first")
        ckpt = checkpoint_path(workdir)
        if not ckpt.is_file():
            raise RuntimeError(f"no checkpoint at {ckpt} - run download_checkpoint() first")
        if log:
            print(f"[ir] {precision}: exporting against {clone}")
        source = "export"
        t0 = time.perf_counter()
        ir_root.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            REPO=str(clone),
            DIFFSYNTH_MODEL_BASE_PATH=str(clone / "checkpoints"),
            PYTHONPATH=str(clone / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        )
        # The interpreter is pinned to sys.executable (the notebook environment); clear any
        # VIRTUAL_ENV this kernel inherited from the shell that launched Jupyter, so the
        # subprocess does not get dragged back into some other project's site-packages.
        env.pop("VIRTUAL_ENV", None)
        ref_dump = make_ref_dump(workdir, log=log)
        for unit in UNITS:
            if log:
                print(f"[ir] {precision}: exporting {unit} ...", flush=True)
            proc = subprocess.run(_export_cmd(unit, precision, ir_root, ref_dump), capture_output=True, text=True, cwd=str(clone), env=env)
            if proc.returncode != 0:
                tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
                raise RuntimeError(f"export of {unit} ({precision}) failed:\n{tail}")
            _write_ref_sidecar(unit)
            # The IR is written; the 18 GB ONNX it was traced through is dead weight from here on.
            # Delete per unit, so the peak never holds the whole set's scratch at once.
            if not keep_onnx:
                _drop_onnx_scratch(ir_root, (unit,), log=log)
        seconds = round(time.perf_counter() - t0, 1)
        if log:
            print(f"[ir] {precision}: exported in {seconds} s")

    rename_ir_files(ir_root, precision)
    paths = ir_paths(ir_root, precision)
    fingerprint = set_fingerprint(paths)
    manifest_path.write_text(
        json.dumps(
            {
                "precision": precision,
                "ir_root": str(ir_root),
                "flag_hash": fingerprint_flags,
                "flags": EXPORT_FLAGS,
                "fingerprint": fingerprint,
                "source": source,
                "seconds": seconds,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    return {"precision": precision, "paths": paths, "fingerprint": fingerprint, "source": source, "seconds": seconds, "cached": source == "cache"}


def ir_size_mb(paths: dict[str, Path]) -> float:
    """Return the total on-disk size of an IR set in megabytes."""
    total = 0
    for xml in paths.values():
        total += xml.stat().st_size + (xml.with_suffix(".bin").stat().st_size if xml.with_suffix(".bin").is_file() else 0)
    return round(total / 1e6, 1)


def free_gb(path: str | Path) -> float:
    """Return the free space in GB on the filesystem holding ``path``."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return round(shutil.disk_usage(probe).free / 1e9, 1)


# --------------------------------------------------------------- the text encoder (umT5-xxl)

# FasterWAM is a world-action model whose first input is language: the instruction is tokenized,
# run through a umT5-xxl encoder and handed to the video and action experts as 128 text tokens (the
# runtime appends one proprio token, hence the 129-token context the graphs take).  The conversion
# of the three graphs does not need that encoder - the shipped artifacts carry the encoded contexts
# of the evaluation set - so it is converted separately here, with its own cache directory, its own
# precision rule (fp16 weights; f32 activations to export and on the CPU, bf16 on a GPU; optional
# INT8_SYM weights for the GPU, see ensure_t5_int8) and its own device rule (it follows the
# pipeline's device, see load_t5_for).
T5_SEQ = 128  # tokens per prompt, baked into the exported graph
T5_DIM = 4096  # width of each token of the context
T5_TAG = "t5"
T5_MIN_COS = 0.9999  # gate against the shipped contexts, over the whole bank
T5_SCRIPT = EXPORT_DIR / "export_t5.py"
T5_INT8_SCRIPT = EXPORT_DIR / "compress_t5.py"
# Weight storages of the encoder.  fp16 is the exported IR (exact on the CPU); int8 is NNCF INT8_SYM
# (data-free, per-channel, all 169 weights) applied to that IR - 5.7 GB instead of 11.4 GB and about 2x
# faster per prompt on an integrated GPU, measured in bf16 execution there: context cos_min 0.96375 /
# mean 0.99583 against the shipped bank (fp16 weights 0.98975 / 0.99918), action chunk cos_min
# 0.999857 / mean 0.999974 against the shipped-context chunk (fp16 weights 0.999920 / 0.999985).
T5_WEIGHTS = ("fp16", "int8")
# The checkpoints were trained on instructions wrapped in this sentence (every row of
# `artifacts/eval_samples.json` is the template plus the task string), so a task typed into the
# notebook is wrapped the same way before it is encoded.
T5_INSTRUCTION_PREFIX = "A video recorded from a robot's point of view executing the following instruction: "
# Execution precisions the encoder can be compiled for.  The storage is the IR (fp16, or int8 - see
# T5_WEIGHTS); this is the arithmetic the device plugin runs it in, and it is the knob that moves this
# encoder's accuracy most.  Measured on an integrated GPU: f32 302 ms/prompt at cos 1.00000000, bf16
# 121.7 ms at cos 0.9748, f16 NaN (the residual stream reaches 3e5, above the f16 maximum 65504).
T5_PRECISIONS = ("f32", "bf16", "f16")
# Both are fetched by the repo's own loader on the first export, into the clone's checkpoints/.
T5_CKPT_REL = "checkpoints/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
T5_TOK_REL = "checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
T5_ONNX_GB = 22.8  # fp32 external data the trace writes, deleted after the IR


def t5_dir(workdir: str | Path, weights: str = "fp16") -> Path:
    """Return the directory the text encoder's IR is WRITTEN to (``<model>/ir/<weights>/t5``).

    The encoder's weight storage is its own choice (``fp16`` = the exact arm, f32 on the CPU; ``int8``
    = ``INT8_SYM`` weights, the fast GPU arm, see :func:`ensure_t5_int8`), which is why it gets a
    directory per weight storage rather than one shared ``model/ir/t5``: ``model/ir/`` holds exactly
    one level of precisions - ``fp16/`` and ``int8/`` - each carrying its four units (``vae``,
    ``video``, ``action``, ``t5``).

    NB the text encoder therefore lives INSIDE the precision directory: deleting ``model/ir/int8``
    deletes the INT8 text encoder too (about 20 s to rebuild), and deleting ``model/ir/fp16`` deletes
    the fp16 one (about 15 min, the ONNX trace is 22.8 GB).  The pre-12-Nov flat ``model/ir/t5/`` is
    still READ as a fallback by :func:`t5_paths` / :func:`t5_manifest_path`.
    """
    return model_dir(workdir) / "ir" / _t5_weights(weights) / T5_TAG


def _t5_legacy_dir(workdir: str | Path) -> Path:
    """The pre-12-Nov home of the text encoder (``<model>/ir/t5``): read-only fallback."""
    return model_dir(workdir) / "ir" / T5_TAG


def _t5_weights(weights: str) -> str:
    weights = str(weights or "fp16").lower()
    if weights not in T5_WEIGHTS:
        raise ValueError(f"weights must be one of {T5_WEIGHTS}, got {weights!r}")
    return weights


def t5_paths(workdir: str | Path, weights: str = "fp16") -> dict[str, Path]:
    """Return ``{"xml", "bin"}`` of the text encoder with ``fp16`` (exported) or ``int8`` weights."""
    w = _t5_weights(weights)
    xml = t5_dir(workdir, w) / f"{T5_TAG}_{w}.xml"
    if not xml.is_file():
        legacy = _t5_legacy_dir(workdir) / f"{T5_TAG}_{w}.xml"
        if legacy.is_file():
            xml = legacy
    return {"xml": xml, "bin": xml.with_suffix(".bin")}


def _t5_manifest(workdir: str | Path, name: str, weights: str) -> Path:
    """Where a text-encoder manifest is read from: the current layout first, the flat one second."""
    new = t5_dir(workdir, weights) / name
    if new.is_file():
        return new
    legacy = _t5_legacy_dir(workdir) / name
    return legacy if legacy.is_file() else new


def t5_int8_manifest_path(workdir: str | Path) -> Path:
    """Return the file the INT8 compression of the text encoder describes itself in."""
    return _t5_manifest(workdir, f"{T5_TAG}_int8_manifest.json", "int8")


def t5_manifest_path(workdir: str | Path) -> Path:
    """Return the file the text encoder's export describes itself in."""
    return _t5_manifest(workdir, f"{T5_TAG}_manifest.json", "fp16")


def t5_assets(workdir: str | Path) -> dict[str, Path]:
    """Return the two files the export needs from Hugging Face (encoder weights, tokenizer).

    Neither ships with the notebook: the encoder is 11.4 GB of bf16 weights (only needed to *make*
    the IR - the IR itself is 11.4 GB of fp16) and the tokenizer is 21 MB.
    """
    clone = source_dir(workdir)
    return {"checkpoint": clone / T5_CKPT_REL, "tokenizer": clone / T5_TOK_REL}


def t5_size_mb(paths: dict[str, Path]) -> float:
    """Return the on-disk size of the exported text encoder in megabytes."""
    total = 0.0
    for path in paths.values():
        if path.is_file():
            total += path.stat().st_size
    return round(total / 1e6, 1)


def t5_export_cmd(workdir: str | Path, keep_onnx: bool = False, free_checkpoint: bool = False) -> list[str]:
    """Return the argv that exports the text encoder for this notebook."""
    wd = Path(workdir).resolve()
    cmd = [
        sys.executable,
        str(T5_SCRIPT),
        "--out-dir",
        str(t5_dir(wd, "fp16")),
        "--samples",
        str(wd / "artifacts" / "eval_samples.json"),
        "--ctx-ref",
        str(wd / "artifacts" / "eval_contexts.npz"),
        "--min-cos",
        str(T5_MIN_COS),
    ]
    if keep_onnx:
        cmd += ["--keep-onnx"]
    if free_checkpoint:
        cmd += ["--free-checkpoint"]
    return cmd


def _run_streaming(cmd: list[str], cwd: str | Path, env: dict, keep_tail: int = 40, tail: list[str] | None = None) -> int:
    """Run a subprocess, echoing its output line by line; return the exit code.

    The text-encoder export takes minutes and downloads 11.4 GB, so its progress is worth watching
    live instead of reading it after a silent ``capture_output`` wait.  The child is killed if the
    read loop is interrupted (a notebook "Interrupt kernel" included), so it never outlives the
    cell.

    Args:
        cmd: the argv.
        cwd: working directory of the child.
        env: its environment.
        keep_tail: how many of the last output lines to keep.
        tail: optional list that receives those last lines, for the caller's error message.
    """
    lines = tail if tail is not None else []
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            del lines[:-keep_tail]
            print(line, flush=True)
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def ensure_t5(
    workdir: str | Path,
    force: bool = False,
    log: bool = True,
    keep_onnx: bool = False,
    free_checkpoint: bool = False,
) -> dict:
    """Return the exported umT5-xxl text encoder, exporting it into ``<model>/ir/fp16/t5`` if needed.

    The encoder turns an instruction into the 128-token context the three graphs start at, which
    is what makes the notebook end-to-end (text in, action chunk out) instead of starting from the
    contexts cached in ``artifacts/eval_contexts.npz``.

    Cost, measured: 11.4 GB of bf16 checkpoint + 22.8 GB of fp32 ONNX scratch + 11.4 GB of IR on
    disk, about 46 GB of RAM at the peak of the trace, and roughly 15 min on 16 cores.  The ONNX
    scratch is deleted as soon as the IR exists (``keep_onnx`` keeps it, ``FWAM_KEEP_ONNX=1`` too).

    Args:
        workdir: the notebook's directory (owns ``model/``).
        force: re-export even when the IR and its manifest are already there.
        log: print one line per decision (the exporter's own output is always streamed).
        keep_onnx: keep the ``<model>/ir/fp16/t5/onnx`` scratch instead of deleting it.
        free_checkpoint: delete the 11.4 GB bf16 checkpoint once the ONNX is written, so a tight
            disk never holds the checkpoint and the IR at the same time (it re-downloads later).

    Returns:
        ``{"paths", "manifest", "source", "seconds"}`` where ``source`` is ``"cache"`` or
        ``"export"`` and ``manifest`` holds the export's own measurements and gate result.
    """
    workdir = Path(workdir).resolve()
    keep_onnx = keep_onnx or os.environ.get("FWAM_KEEP_ONNX", "").strip().lower() in ("1", "true", "yes")
    paths = t5_paths(workdir)
    manifest: dict = {}
    if t5_manifest_path(workdir).is_file():
        try:
            manifest = json.loads(t5_manifest_path(workdir).read_text())
        except json.JSONDecodeError:
            manifest = {}
    complete = paths["xml"].is_file() and paths["bin"].is_file()
    # export_t5.py writes the manifest after the IR and the gate, and exits non-zero on a failed
    # gate: no manifest means the export never finished, and a failed gate must not be reused.
    if complete and not force and not manifest:
        if log:
            print(f"[t5] {paths['xml'].name} has no manifest (an interrupted export?) - exporting again")
        complete = False
    if complete and not force and not manifest.get("gate", {}).get("passed", False):
        gate = manifest.get("gate", {})
        raise RuntimeError(
            f"the text encoder at {paths['xml']} failed its accuracy gate "
            f"(cos_min {gate.get('cos_min')}) - delete {paths['xml'].parent} or "
            f"pass force=True"
        )
    if complete and not force:
        if log:
            gate = manifest.get("gate", {})
            print(
                f"[t5] reusing {paths['xml']} ({t5_size_mb(paths) / 1000:.2f} GB"
                + (f", context cos_min {gate['cos_min']:.9f}" if gate.get("cos_min") else "")
                + ")"
            )
        return {"paths": paths, "manifest": manifest, "source": "cache", "seconds": 0.0}

    clone = source_dir(workdir)
    if not (clone / "src" / "fasterwam").is_dir():
        raise RuntimeError(f"no FasterWAM source clone at {clone} - run clone_source() first")
    for name in ("eval_samples.json", "eval_contexts.npz"):
        if not (workdir / "artifacts" / name).is_file():
            raise RuntimeError(f"missing artifacts/{name} - the export gates against the shipped contexts")

    if log:
        assets = t5_assets(workdir)
        have = assets["checkpoint"].is_file()
        print(f"[t5] exporting umT5-xxl -> {t5_dir(workdir, 'fp16')}")
        print(f"[t5]   checkpoint {'present' if have else 'to download (11.4 GB)'}: {assets['checkpoint']}")
        print(f"[t5]   tokenizer  {'present' if (assets['tokenizer'] / 'tokenizer.json').is_file() else 'to download (21 MB)'}: {assets['tokenizer']}")
        print(f"[t5]   needs ~{T5_ONNX_GB + 11.4:.1f} GB free during the trace; {free_gb(workdir):.1f} GB free now")

    env = dict(
        os.environ,
        REPO=str(clone),
        DIFFSYNTH_MODEL_BASE_PATH=str(clone / "checkpoints"),
        PYTHONPATH=str(clone / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    env.pop("VIRTUAL_ENV", None)
    t0 = time.perf_counter()
    tail: list[str] = []
    code = _run_streaming(t5_export_cmd(workdir, keep_onnx=keep_onnx, free_checkpoint=free_checkpoint), cwd=clone, env=env, tail=tail)
    seconds = round(time.perf_counter() - t0, 1)
    if t5_manifest_path(workdir).is_file():
        manifest = json.loads(t5_manifest_path(workdir).read_text())
    if code != 0:
        gate = manifest.get("gate", {})
        raise RuntimeError(
            f"text encoder export failed (exit {code})" + (f": context cos_min {gate.get('cos_min')}" if gate else "") + "\n" + "\n".join(tail[-15:])
        )
    if log:
        print(f"[t5] exported in {seconds} s")
    return {"paths": t5_paths(workdir), "manifest": manifest, "source": "export", "seconds": seconds}


def t5_int8_cmd(workdir: str | Path) -> list[str]:
    """Return the argv that compresses the fp16 text encoder to INT8_SYM weights."""
    wd = Path(workdir).resolve()
    return [sys.executable, str(T5_INT8_SCRIPT), "--src", str(t5_paths(wd)["xml"]), "--out-dir", str(t5_dir(wd, "int8"))]


def ensure_t5_int8(workdir: str | Path, force: bool = False, log: bool = True) -> dict:
    """Return the INT8_SYM text encoder, compressing the fp16 IR into ``<model>/ir/int8/t5/t5_int8.xml``
    if needed.

    NNCF ``compress_weights(mode=INT8_SYM)``: data-free, per-channel, every weight including the token
    embedding.  The fp16 IR is only read, never changed, so both arms stay on disk (11.4 + 5.7 GB).
    Cost, measured: about 17 s and about 17 GB of host memory, in a subprocess so the memory is
    released when it ends.  Measured in bf16 execution on an integrated GPU: 48 ms per prompt against
    91-98 ms for fp16 weights, context cos_min 0.96375 / mean 0.99583 against the shipped bank (fp16:
    0.98975 / 0.99918), action chunk cos_min 0.999857 against the shipped-context chunk (fp16: 0.999920).

    Args:
        workdir: the notebook's directory (owns ``model/``).
        force: compress again even when the INT8 IR and its manifest are already there.
        log: print one line per decision (the compressor's own output is always streamed).

    Returns:
        ``{"paths", "manifest", "source", "seconds"}`` where ``source`` is ``"cache"`` or ``"compress"``.

    Raises:
        RuntimeError: when the fp16 IR is missing (run :func:`ensure_t5` first) or the compression fails.
    """
    workdir = Path(workdir).resolve()
    src = t5_paths(workdir)
    if not (src["xml"].is_file() and src["bin"].is_file()):
        raise RuntimeError(f"no fp16 text encoder at {src['xml']} - run ensure_t5() first")
    paths = t5_paths(workdir, "int8")
    manifest_path = t5_int8_manifest_path(workdir)  # cache check: current layout, else the flat one
    written = t5_dir(workdir, "int8") / f"{T5_TAG}_int8_manifest.json"  # what a run writes
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    # compress_t5.py writes the manifest last: no manifest means an interrupted run, so do it again.
    if paths["xml"].is_file() and paths["bin"].is_file() and manifest and not force:
        if log:
            print(f"[t5] reusing {paths['xml']} ({t5_size_mb(paths) / 1000:.2f} GB, " f"{manifest.get('recipe', {}).get('mode', '?')} weights)")
        return {"paths": paths, "manifest": manifest, "source": "cache", "seconds": 0.0}

    if log:
        print(
            f"[t5] compressing {src['xml'].name} -> {paths['xml'].name} (NNCF INT8_SYM, data-free; "
            f"about 17 GB of host memory for about 20 s; {free_gb(workdir):.1f} GB of disk free)"
        )
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    t0 = time.perf_counter()
    tail: list[str] = []
    code = _run_streaming(t5_int8_cmd(workdir), cwd=workdir, env=env, tail=tail)
    seconds = round(time.perf_counter() - t0, 1)
    if code != 0 or not written.is_file():
        raise RuntimeError(f"text encoder INT8 compression failed (exit {code})\n" + "\n".join(tail[-15:]))
    manifest = json.loads(written.read_text())
    if log:
        print(f"[t5] compressed in {seconds} s")
    return {"paths": paths, "manifest": manifest, "source": "compress", "seconds": seconds}


def t5_prompt(task: str) -> str:
    """Wrap a task string in the instruction template the checkpoints were trained on.

    The exported graph is fed the templated sentence, not the bare task: every row of
    ``artifacts/eval_samples.json`` is ``T5_INSTRUCTION_PREFIX + task``, and the shipped contexts
    were produced from those rows.  Encoding a bare task would hand the encoder an out-of-
    distribution prompt (the context cosine drops to 0.06), so the notebook always wraps.

    Args:
        task: the instruction as a human would say it, e.g. "pick up the black bowl".

    Returns:
        The sentence to tokenize.
    """
    task = str(task).strip()
    if task.startswith(T5_INSTRUCTION_PREFIX):
        return task
    return T5_INSTRUCTION_PREFIX + task


def t5_tokenizer(workdir: str | Path):
    """Return the production tokenizer (the repo's own wrapper, ``clean="whitespace"``, 128 tokens).

    Using the repo class rather than a bare ``AutoTokenizer`` matters: it applies the same text
    cleaning the shipped contexts were built with, and it pads/truncates to the graph's 128 tokens.
    """
    tok_dir = t5_assets(workdir)["tokenizer"]
    if not tok_dir.is_dir():
        raise RuntimeError(f"no tokenizer at {tok_dir} - run ensure_t5() first")
    src = source_dir(workdir) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from fasterwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    return HuggingfaceTokenizer(name=str(tok_dir), seq_len=T5_SEQ, clean="whitespace")


def t5_config(device: str, precision: str = "f32") -> dict:
    """Return the ``compile_model`` properties that pin the encoder's *execution* precision.

    The IR's weights (fp16 or int8) are not touched here; this is the arithmetic the device plugin
    runs it in.  Two measured facts make the hint necessary:

    * an Intel GPU defaults to fp16 execution and this graph **overflows** there (24 baked
      ``finfo.min`` constants become ``-inf`` and the residual stream reaches 3e5 against an f16
      maximum of 65504), so the outputs come back NaN - ``f32``/``bf16`` have to be stated;
    * the CPU already runs in f32, and leaving its default alone reproduces the arm that was gated
      (context cos 0.99999992), so no hint is passed there.
    """
    precision = str(precision or "f32").lower()
    if precision not in T5_PRECISIONS:
        raise ValueError(f"precision must be one of {T5_PRECISIONS}, got {precision!r}")
    if device.upper().startswith("CPU") and precision == "f32":
        return {}
    return {"INFERENCE_PRECISION_HINT": precision}


class TextEncoder:
    """The umT5-xxl encoder as the notebook uses it: instruction in, 128-token context out.

    One prompt per call (the graph is static ``[1, 128]``), fp16 or INT8_SYM weights, and whatever device
    and execution precision the caller asks for - see :func:`load_t5` for the measured arms.  f32 execution
    is the exact arm (context cos 0.9999+); bf16 rounds the context to 0.9748 on the worst prompt but
    is what fits a GPU beside the three graphs (see :func:`load_t5_for`); f16 execution NaNs on a GPU.
    INT8_SYM weights in bf16 halve the GPU time per prompt at context cos_min 0.964.

    Attributes:
        unit: the compiled :class:`Unit` (weights in MB, compile time in seconds).
        tokenizer: the repo's ``HuggingfaceTokenizer``.
        device: the device the encoder runs on.
        precision: the execution precision of that device (``f32`` / ``bf16`` / ``f16``).
        weights: the weight storage of the IR (``fp16`` / ``int8``).
        times: per-prompt wall time in ms, appended on every :meth:`encode`.
    """

    def __init__(self, xml: str | Path, tokenizer, device: str = "CPU", precision: str = "f32", threads: int | None = None, weights: str = "fp16") -> None:
        import openvino as ov

        self.unit = Unit(ov.Core(), Path(xml), device, threads=threads, config=t5_config(device, precision), preprocess=False)
        self.tokenizer = tokenizer
        self.device = device
        self.precision = precision
        self.weights = _t5_weights(weights)
        self.times: list[float] = []

    def encode(self, prompts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode one or more instructions.

        Args:
            prompts: a string or a list of strings.

        Returns:
            ``(context [N,128,4096] float32, ids [N,128] int64, mask [N,128] int64)`` -- the mask is
            the true padding mask (the shipped ``eval_contexts.npz`` only kept the all-ones one).

        Raises:
            RuntimeError: when the context is not finite, which is what f16 execution of this
                graph produces on a GPU (overflow); it is never passed on to the graphs.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        ids, mask = self.tokenizer(list(prompts), return_mask=True, add_special_tokens=True)
        ids = np.asarray(ids, dtype=np.int64)
        mask = np.asarray(mask, dtype=np.int64)
        out = []
        for i in range(ids.shape[0]):
            t0 = time.perf_counter()
            out.append(self.unit(ids[i : i + 1], mask[i : i + 1])[0])
            self.times.append((time.perf_counter() - t0) * 1e3)
        context = np.concatenate(out, axis=0).astype(np.float32)
        if not np.isfinite(context).all():
            raise RuntimeError(
                f"the text encoder returned a non-finite context on {self.device} " f"({self.precision} execution) - f16 overflows this graph; use bf16 or f32"
            )
        return context, ids, mask

    def __call__(self, prompt: str) -> np.ndarray:
        """Encode one instruction into ``[1, 128, 4096]``."""
        return self.encode([prompt])[0]

    @property
    def ms_per_prompt(self) -> float:
        """Mean wall time of the last up-to-5 prompts, in ms."""
        return round(float(np.mean(self.times[-5:])), 1) if self.times else 0.0

    @property
    def label(self) -> str:
        """``"<device> <precision>"``, plus ``", int8 weights"`` for the INT8 arm - for printouts."""
        return f"{self.device} {self.precision}" + (", int8 weights" if self.weights == "int8" else "")


def load_t5(
    workdir: str | Path, device: str = "CPU", precision: str = "f32", threads: int | None = None, log: bool = True, weights: str = "fp16"
) -> TextEncoder:
    """Compile the text encoder for a device, an execution precision and a weight storage.

    ``weights`` picks the IR (fp16 = exported, int8 = :func:`ensure_t5_int8`); ``precision`` the
    arithmetic and ``device`` the device.  Where it matters, measured on an integrated GPU over the 32
    LIBERO instructions (cosine against the shipped torch bank, which is exact fp32):

    ======  =======  =============  ===========  ==========
    device  weights  precision      ms/prompt    cos_min
    ======  =======  =============  ===========  ==========
    CPU     fp16     f32            1480         0.99999992
    CPU     fp16     bf16           1430         0.99793514
    GPU     fp16     f32            302          1.00000000
    GPU     fp16     bf16           91-122       0.97478878
    GPU     int8     bf16           48           0.96375
    GPU     fp16     f16            109          nan (overflow)
    ======  =======  =============  ===========  ==========

    GPU f32 is exact and 4.9x the CPU arm, but needs ~21 GB of device memory, which does not fit beside
    the three graphs; ``bf16`` is 2.5x faster again, rounds the context, and is what
    :func:`load_t5_for` picks on a GPU.  INT8_SYM weights halve that again (and the memory); the
    context drops to cos_min 0.964 / mean 0.9958, but the action chunk stays at cos_min 0.999857
    against the shipped-context chunk (fp16 weights: 0.999920).  ``f16`` must never be used: an
    Intel GPU defaults to it and this graph overflows (24 baked ``finfo.min`` constants, residuals up
    to 3e5 against an f16 maximum of 65504) - the hint is what stops that.

    Args:
        workdir: the notebook's directory (owns ``model/ir/<weights>/t5``).
        device: ``CPU`` or any GPU device string ``core.available_devices`` reports.
        precision: ``f32`` (default, exact), ``bf16`` (fast, rounded) or ``f16`` (overflows on GPU).
        threads: CPU inference threads; ``None`` (default) leaves the choice to the CPU plugin.
            Ignored on a GPU.
        log: print one line with the compiled size, time and precision.
        weights: ``fp16`` (default, the exported IR) or ``int8`` (INT8_SYM, see :func:`ensure_t5_int8`).

    Returns:
        A :class:`TextEncoder`.
    """
    paths = t5_paths(workdir, weights)
    if not paths["xml"].is_file():
        make = "ensure_t5_int8()" if weights == "int8" else "ensure_t5()"
        raise FileNotFoundError(f"no text encoder at {paths['xml']} - run {make} first")
    encoder = TextEncoder(paths["xml"], t5_tokenizer(workdir), device=device, precision=precision, threads=threads, weights=weights)
    if log:
        print(
            f"[t5] {device} {encoder.precision} execution, {encoder.weights} weights: "
            f"weights={encoder.unit.weight_mb / 1000:.2f} GB compile={encoder.unit.compile_s:.1f} s"
        )
    return encoder


def load_t5_for(workdir: str | Path, device: str = "CPU", log: bool = True, weights: str = "fp16") -> TextEncoder:
    """Compile the text encoder the way the notebook wants it: following the pipeline's device.

    On a GPU it runs in bf16 (13x the CPU arm, and its 11.4 GB of fp16 weights fit beside the three
    graphs); a GPU that cannot do bf16 falls back to the exact CPU arm rather than to f32-on-GPU, which
    needs ~21 GB of device memory and makes the graph compile fail with ``CL_OUT_OF_RESOURCES``.  When
    the pipeline runs on the CPU, the encoder runs there too.  ``weights="int8"`` applies to the GPU arm
    only: the CPU arm (chosen or fallen back to) always runs the fp16 IR in f32, the exact one.

    "Cannot do bf16" is decided twice: up front, when the device's ``OPTIMIZATION_CAPABILITIES`` do
    not list ``BF16``, and afterwards, when compiling raises an OpenVINO ``RuntimeError`` (typically
    device memory still held by an earlier encoder or graph set).  Any other exception - a missing
    module, a bad argument, a missing IR - is a bug, not a device limit, and propagates.

    Args:
        workdir: the notebook's directory (owns ``model/ir/<weights>/t5``).
        device: the pipeline's device, i.e. what the "Run on" dropdown picked.
        log: print one line per decision.
        weights: ``fp16`` (default) or ``int8`` for the GPU arm (run :func:`ensure_t5_int8` first).

    Returns:
        A :class:`TextEncoder`; its ``.device`` / ``.precision`` / ``.weights`` say which arm was taken.
    """
    weights = _t5_weights(weights)
    if str(device).upper().startswith("GPU"):
        if not t5_paths(workdir, weights)["xml"].is_file():
            make = "ensure_t5_int8()" if weights == "int8" else "ensure_t5()"
            raise FileNotFoundError(f"no text encoder at {t5_paths(workdir, weights)['xml']} - run {make} first")
        reason = None
        try:
            import openvino as ov

            caps = [str(c).upper() for c in ov.Core().get_property(device, "OPTIMIZATION_CAPABILITIES")]
            if "BF16" not in caps:
                reason = f"{device} does not report BF16 among its capabilities {caps}"
        except Exception:  # noqa: BLE001
            pass  # unknown: try it and let compile decide
        if reason is None:
            try:
                return load_t5(workdir, device=device, precision="bf16", log=log, weights=weights)
            except FileNotFoundError:
                raise
            except RuntimeError as exc:
                reason = f"{device} cannot run the encoder in bf16 ({type(exc).__name__}): {str(exc)[:120]}"
                if log:
                    print(f"[t5] {reason}")
                    print(
                        "[t5] if that is device memory (a previous encoder or graph set still resident), " "release it or restart the kernel to get bf16 back"
                    )
        elif log:
            print(f"[t5] {reason}")
        if log:
            print("[t5] falling back to the CPU arm - exact, and about 13x slower per prompt")
    if weights != "fp16" and log:
        print(f"[t5] {weights} weights are a GPU option: the CPU arm runs the fp16 IR in f32 (the exact arm)")
    return load_t5(workdir, device="CPU", precision="f32", log=log)


def with_text_context(inp: dict, context: np.ndarray) -> dict:
    """Swap the 128 text tokens of an input dict for freshly encoded ones, keeping its tail tokens.

    ``eval_inputs`` hands the graphs a 129-token context: 128 text tokens plus the proprio token
    the runtime appends.  Replacing only the first 128 keeps the proprio projection exactly as the
    artifacts had it, so a T5-encoded run and a bank-context run differ in nothing but the language.

    The mask over those 128 tokens is **all ones**, because that is what the model itself passes:
    ``encode_prompt`` zeroes the context past ``mask.sum()`` and then returns an all-ones mask, and
    the shipped bank carries that all-ones mask.  Feeding the tokenizer's true padding mask instead
    is a different graph input, not a correction - it changes the action chunk completely (measured:
    cos -0.09 against the bank-context chunk, against 1.000000000 with the all-ones mask).

    Args:
        inp: a dict from :func:`eval_inputs` or :func:`sample_inputs`.
        context: exactly ``[1, 128, 4096]`` (or ``[128, 4096]``), as returned by :class:`TextEncoder`.

    Returns:
        A shallow copy of ``inp`` with ``context`` and ``context_mask`` replaced.

    Raises:
        ValueError: when ``context`` is not one prompt of ``T5_SEQ`` (128) tokens of the graph's
            width, or ``inp`` has no text part of that size to replace.
    """
    out = dict(inp)
    old = np.asarray(inp["context"], dtype=np.float32)
    new = np.asarray(context, dtype=np.float32)
    if new.ndim == 2:
        new = new[None]
    if new.shape != (1, T5_SEQ, T5_DIM):
        raise ValueError(f"context must be [1, {T5_SEQ}, {T5_DIM}] (one encoded prompt), got {list(new.shape)}")
    if old.ndim != 3 or old.shape[0] != 1 or old.shape[1] < T5_SEQ or old.shape[2] != T5_DIM:
        raise ValueError(f"inp['context'] must be [1, >={T5_SEQ}, {T5_DIM}], got {list(old.shape)}")
    head_mask = np.ones((1, T5_SEQ), np.float32)
    old_mask = np.asarray(inp["context_mask"], dtype=np.float32)
    out["context"] = np.concatenate([new, old[:, T5_SEQ:]], axis=1).astype(np.float32)
    out["context_mask"] = np.concatenate([head_mask, old_mask[:, T5_SEQ:]], axis=1).astype(np.float32)
    return out


# --------------------------------------------------------------------------- torch reference


def torch_ref_path(workdir: str | Path) -> Path:
    """Return the file the on-CPU PyTorch reference chunk is cached in."""
    return Path(workdir) / "artifacts" / TORCH_REF_NAME


# The torch reference runs in float32 only when this much RAM is available (fp32 weights alone are
# ~25 GB, plus the page cache of the checkpoint); torch_ref_actions.py applies the same rule.
TORCH_REF_FP32_GIB = 40


def available_gib() -> float:
    """Return the RAM a new process could get, in GiB (``MemAvailable``, or the raw free pages)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (AttributeError, ValueError, OSError):  # Windows: no sysconf
        return 0.0


def torch_ref_identity(pack: str | Path, ckpt: str | Path) -> dict:
    """Return what a cached torch reference was computed from: the input pack and the checkpoint.

    The pack is small, so it is identified by content (md5); the 11 GB checkpoint by its size.
    """
    pack, ckpt = Path(pack), Path(ckpt)
    return {
        "pack_md5": hashlib.md5(pack.read_bytes()).hexdigest() if pack.is_file() else None,
        "ckpt_bytes": ckpt.stat().st_size if ckpt.is_file() else None,
    }


def _identity_differs(recorded: dict, current: dict) -> bool:
    """Return True when a field known on both sides differs (an unknown field never invalidates)."""
    return any(recorded.get(k) is not None and current.get(k) is not None and recorded[k] != current[k] for k in current)


def ensure_torch_ref(workdir: str | Path, steps: int = 10, dtype: str = "auto", force: bool = False, log: bool = True) -> dict:
    """Return the torch-on-the-CPU reference chunk for the regression pack, computing it if needed.

    The kernel never imports torch: this runs ``torch_ref_actions.py`` as a subprocess, the same way
    the export runs (the CPU wheel is installed, the notebook itself is OpenVINO only).  The script
    rebuilds the model from the clone and the checkpoint and replays the three graphs eagerly, then
    writes ``artifacts/torch_ref_pack.npz``; that file is reused afterwards, because the run is
    expensive - about 30 s once the checkpoint is in the page cache, minutes on the first run.

    Args:
        workdir: the notebook's directory (owns ``model/`` and ``artifacts/``).
        steps: Euler steps of the action loop; a cached reference made with other ``steps`` is
            rebuilt, since the chunk is only comparable at the same step count.  So is one made from
            a different input pack or checkpoint (see :func:`torch_ref_identity`); a cache written
            before that identity was recorded is trusted.
        dtype: ``"auto"`` (float32 when the host has the RAM for it, else bfloat16 - an fp32 model
            is 25 GB of weights alone), ``"float32"`` or ``"bfloat16"``.  With ``"auto"`` a cached
            bfloat16 reference is reused even when the host could now run float32; one line says so.
        force: recompute even when a cached reference is present.
        log: forward the subprocess output as it arrives.

    Returns:
        The cache file as a dict: ``action [32,7]``, ``first_frame``, ``kv [16,1,294,3072]``,
        ``dtype``, ``steps``, ``timing`` and ``checks`` (both decoded from JSON).  ``checks`` holds
        the script's own agreement with the fp32 reference the pack ships, which is how a broken
        replay shows up instead of silently shifting every number.
    """
    workdir = Path(workdir).resolve()
    out = torch_ref_path(workdir)
    pack = artifacts_dir(workdir) / "ov_runtime_pack.npz"

    def _decode(cache: dict) -> dict:
        for key in ("timing", "checks", "identity"):
            if isinstance(cache.get(key), str):
                cache[key] = json.loads(cache[key])
        return cache

    cached = None
    if out.is_file() and not force:
        with np.load(out) as z:
            cached = _decode({k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files})
        recorded = cached.get("identity")
        if int(cached.get("steps", -1)) != int(steps):
            if log:
                print(f"[torch] cached reference is for {cached.get('steps')} steps, not {steps} - rebuilding")
            cached = None
        elif dtype != "auto" and cached.get("dtype") != dtype:
            if log:
                print(f"[torch] cached reference is {cached.get('dtype')}, not {dtype} - rebuilding")
            cached = None
        elif recorded and _identity_differs(recorded, torch_ref_identity(pack, checkpoint_path(workdir))):
            if log:
                print("[torch] cached reference was computed from another input pack or checkpoint - rebuilding")
            cached = None
    if cached is not None:
        if log:
            print(f"[torch] reusing the reference at {out.relative_to(workdir)} " f"({cached['dtype']}, chunk {cached['timing']['total_s']:.1f} s on the CPU)")
            if dtype == "auto" and cached.get("dtype") == "bfloat16" and available_gib() >= TORCH_REF_FP32_GIB:
                print(f"[torch] this host now has {available_gib():.0f} GiB available: delete {out.name} " f"to recompute it as the exact float32 reference")
        return cached

    if not TORCH_REF_SCRIPT.is_file():
        raise FileNotFoundError(f"missing {TORCH_REF_SCRIPT}")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("VIRTUAL_ENV", None)  # keep the subprocess in this notebook's environment
    cmd = [sys.executable, str(TORCH_REF_SCRIPT), "--workdir", str(workdir), "--pack", str(pack), "--out", str(out), "--dtype", dtype, "--steps", str(steps)]
    if log:
        print("[torch] building the reference chunk with PyTorch on the CPU " "(this is the slow part; the result is cached)", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(workdir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    try:
        for line in proc.stdout:  # stream progress into the notebook
            line = line.rstrip("\r\n")
            if not line or line.startswith("SUMMARY ") or "it/s]" in line:
                continue
            tail.append(line)
            if log:
                print(f"  {line}", flush=True)
        code = proc.wait()
    finally:
        if proc.poll() is None:  # interrupted: never orphan the child
            proc.kill()
            proc.wait()
    if code != 0 or not out.is_file():
        raise RuntimeError("the torch reference failed:\n" + "\n".join(tail[-25:]))

    with np.load(out) as z:
        ref = _decode({k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files})
    return ref


# --------------------------------------------------------------------------- inference


def device_choices() -> tuple[list[tuple[str, str]], str]:
    """List the devices inference can run on, and pick a default.

    Returns:
        ``(options, default)``: ``options`` is ``[(label, device), ...]`` for the CPU and every GPU
        OpenVINO reports (``label`` carries the device's full name); ``default`` is a discrete GPU
        if there is one, else any GPU, else ``"CPU"``.

    Raises:
        RuntimeError: when OpenVINO sees neither a CPU nor a GPU.
    """
    import openvino as ov

    core = ov.Core()
    devices = [d for d in core.available_devices if d == "CPU" or d.startswith("GPU")]
    if not devices:
        raise RuntimeError("OpenVINO sees neither a CPU nor a GPU device")

    def name(device: str) -> str:
        try:
            return core.get_property(device, "FULL_DEVICE_NAME")
        except RuntimeError:
            return "n/a"

    def discrete(device: str) -> bool:
        try:
            return "DISCRETE" in str(core.get_property(device, "DEVICE_TYPE")).upper()
        except RuntimeError:
            return False

    gpus = [d for d in devices if d.startswith("GPU")]
    default = next((d for d in gpus if discrete(d)), gpus[0] if gpus else "CPU")
    return [(f"{d} - {name(d)}", d) for d in devices], default


def _prec_name(ov_type: str) -> str:
    """Short element-type name from a string like ``"<Type: 'float16'>"`` or ``"<Type: 'int64_t'>"``.

    Newer OpenVINO builds report ``int64_t``, older ones ``i64``; the ``_t`` suffix is stripped so
    both read the same.
    """
    name = ov_type.split("'")[-2] if "'" in ov_type else ov_type
    return name[: -len("_t")] if name.endswith("_t") else name


def dtype_for(ov_type: str) -> type:
    """Map an OpenVINO element type string such as ``"<Type: 'float16'>"`` to a numpy dtype.

    Covers the element types the graphs here take (f16/f32 tensors, i64 token ids); anything else is
    rejected rather than silently converted.
    """
    name = _prec_name(ov_type)
    table = {
        "f16": np.float16,
        "float16": np.float16,
        "f32": np.float32,
        "float32": np.float32,
        "f64": np.float64,
        "float64": np.float64,
        "i64": np.int64,
        "int64": np.int64,
        "i32": np.int32,
        "int32": np.int32,
    }
    if name not in table:
        raise ValueError(f"unsupported graph element type {ov_type!r}")
    return table[name]


def as_input(precision: str, array: np.ndarray) -> np.ndarray:
    """Cast ``array`` to a graph input's element type, as a contiguous array."""
    return np.ascontiguousarray(np.asarray(array).astype(dtype_for(precision), copy=False))


def _preprocessed(model):
    """Run OpenVINO's PrePostProcessor over a freshly read model with a no-op request (every input keeps its own element type).

    ``PrePostProcessor.build()`` always runs a graph clean-up on the model (constant / shape-subgraph folding), which
    ``read_model`` + ``compile_model`` alone does not: the video graph goes from 12758 to 11651 ops and its GPU time
    from about 119 to 110 ms, with the same inputs and outputs.  ``benchmark_app`` gets this for free because it builds a
    PrePostProcessor for its input layout; a plain ``compile_model`` call does not.  Set ``FWAM_PPP=0`` to compile the raw model.
    """
    if os.environ.get("FWAM_PPP", "1").strip().lower() in ("0", "false", "no", "off"):
        return model
    from openvino.preprocess import PrePostProcessor

    ppp = PrePostProcessor(model)
    for i, inp in enumerate(model.inputs):
        ppp.input(i).tensor().set_element_type(inp.get_element_type())
    return ppp.build()


class Unit:
    """One compiled OpenVINO model, fed by input position and read by output index.

    Args:
        core: an ``openvino.Core`` instance.
        xml: the IR ``.xml`` file.
        device: any string ``core.available_devices`` reports, for example ``GPU``.
        threads: CPU inference threads (``INFERENCE_NUM_THREADS``); ``None`` (default) leaves the
            choice to the CPU plugin.  Ignored on a GPU.
        config: extra ``compile_model`` properties, for example
            ``{"INFERENCE_PRECISION_HINT": "bf16"}``.  Merged over the defaults.
        preprocess: run :func:`_preprocessed` over the model before compiling (default).  The text encoder passes ``False``: the
            clean-up changes its bf16 numerics (context cos_min 0.990 -> 0.964 over the 32 prompts) and does not make it faster.

    Attributes:
        in_names: input tensor names in binding order.
        in_prec: input element types in binding order.
        in_shapes: static input shapes.
        weight_mb: size of the ``.bin`` weights file.
        compile_s: wall time of ``compile_model``.
        config: the extra compile properties this model was built with.
    """

    def __init__(self, core, xml: str | Path, device: str, threads: int | None = None, config: dict | None = None, preprocess: bool = True) -> None:
        self.xml = Path(xml)
        model = core.read_model(str(self.xml))
        self.model = _preprocessed(model) if preprocess else model
        options = {"INFERENCE_NUM_THREADS": int(threads)} if threads is not None and device.upper().startswith("CPU") else {}
        self.config = dict(config or {})
        options.update(self.config)
        t0 = time.perf_counter()
        self.compiled = core.compile_model(self.model, device, options)
        self.compile_s = round(time.perf_counter() - t0, 2)
        self.core = core
        self.device = device
        self.in_names = [i.get_any_name() for i in self.compiled.inputs]
        self.in_prec = [str(i.get_element_type()) for i in self.compiled.inputs]
        self.out_prec = [str(o.get_element_type()) for o in self.compiled.outputs]
        self.in_shapes = [list(map(int, i.get_shape())) for i in self.compiled.inputs]
        self._dev_inputs = None  # DeviceConstInputs when kvdev is on (see load_units)
        self._async_prefill = None  # PrefillAsync when overlap is on (see load_units)
        weight = self.xml.with_suffix(".bin")
        self.weight_mb = round(weight.stat().st_size / 1e6, 1) if weight.is_file() else 0.0

    def __call__(self, *tensors: np.ndarray) -> list[np.ndarray]:
        """Run one inference with inputs bound by position.

        Args:
            *tensors: one array per graph input, in the order of :attr:`in_names`.

        Returns:
            The output tensors as numpy arrays, in graph order.
        """
        if len(tensors) != len(self.in_prec):
            raise RuntimeError(f"{self.xml.name}: fed {len(tensors)} of {len(self.in_prec)} inputs")
        result = self.compiled([as_input(prec, np.asarray(t)) for prec, t in zip(self.in_prec, tensors)])
        return [np.asarray(result[j]) for j in range(len(self.compiled.outputs))]


class DeviceConstInputs:
    """One persistent InferRequest whose constant-per-chunk inputs live in DEVICE memory (``kvdev``).

    A GPU re-reads a *graph input* at only
    ~26 GB/s, so re-binding the 16 video K/V (29 MB here) as host blobs on all 10 Euler steps costs
    a few hundred microseconds per step.  As ``RemoteContext`` tensors filled once per chunk with
    ``copy_from``, the same bytes stay on the device and each step only re-binds the two that
    actually change (the action latent and t).

    Numerics are bit-identical: only the memory domain of the inputs changes, the graph and the
    exported bytes do not.  Anything unexpected raises from ``__init__`` so the caller can fall back
    to the plain host path.  Inputs bind BY NAME via ``set_tensors`` (the positional
    ``set_input_tensor`` rejects a RemoteTensor, and ``set_tensors`` merges, so the two per-step
    inputs re-bind without disturbing the device-resident ones).
    """

    HOST_IDX = (0, 1)  # latents_action, t -- these change on every Euler step

    def __init__(self, unit: Unit) -> None:
        import openvino as ov

        t0 = time.perf_counter()
        dev = str(unit.device)
        if not dev or dev.upper().startswith(("CPU", "AUTO", "HETERO", "MULTI")):
            raise RuntimeError(f"device {dev!r} has no usable remote context")
        n_in = len(unit.in_prec)
        if n_in < len(self.HOST_IDX) + 2:
            raise RuntimeError(f"expected a K/V-fed action graph, got {n_in} inputs")
        dev_idx = [i for i in range(n_in) if i not in self.HOST_IDX]
        if dev_idx != list(range(max(self.HOST_IDX) + 1, n_in)):
            raise RuntimeError(f"constant inputs are not contiguous after {self.HOST_IDX}: {dev_idx}")
        for i in dev_idx:
            if _prec_name(unit.in_prec[i]) not in ("f16", "half", "float16"):
                raise RuntimeError(f"input {i} ({unit.in_names[i]}) is {unit.in_prec[i]}, expected f16")
            shape = unit.in_shapes[i]
            if len(shape) != 3 or min(shape) <= 0:
                raise RuntimeError(f"input {i} ({unit.in_names[i]}) shape {shape} is not a static 3-D shape")
        if len(unit.compiled.outputs) != 1:
            raise RuntimeError(f"expected 1 output, got {len(unit.compiled.outputs)}")
        if _prec_name(unit.out_prec[0]) not in ("f16", "half", "float16"):
            raise RuntimeError(f"output is {unit.out_prec[0]}, expected f16")
        self.dev_idx = dev_idx
        self.const_names = [unit.in_names[i] for i in dev_idx]
        self.host_names = [unit.in_names[i] for i in self.HOST_IDX]
        self.host_prec = [unit.in_prec[i] for i in self.HOST_IDX]
        self.nbytes = int(sum(int(np.prod(unit.in_shapes[i])) * 2 for i in dev_idx))
        self.ctx = unit.core.get_default_context(dev)  # OCL_CONTEXT / OCL_QUEUE params
        self.req = unit.compiled.create_infer_request()  # ONE request, reused for all steps
        self.dtensors = [self.ctx.create_tensor(ov.Type.f16, ov.Shape(unit.in_shapes[i]), {}) for i in dev_idx]  # keep refs alive for the request
        self.stage = [ov.Tensor(np.zeros(tuple(unit.in_shapes[i]), dtype=np.float16)) for i in dev_idx]
        self.setup_s = round(time.perf_counter() - t0, 3)

    def upload(self, consts) -> float:
        """Land the constant-per-chunk inputs on the device and bind everything. -> ms"""
        if len(consts) != len(self.dev_idx):
            raise RuntimeError(f"got {len(consts)} constants for {len(self.dev_idx)} device inputs")
        return self.upload_part(list(enumerate(consts)))

    def upload_part(self, pairs) -> float:
        """Subset upload: ``pairs`` = [(index into const_names, array), ...] -> ms.

        ``set_tensors`` merges, so a partial bind leaves every earlier binding intact -- that is what
        lets the context-side tensors land while the video prefill is still in flight.
        """
        t0 = time.perf_counter()
        bind = {}
        for j, arr in pairs:
            self.stage[j].data[:] = np.ascontiguousarray(arr)  # host -> host, no re-malloc
            self.dtensors[j].copy_from(self.stage[j])  # the one host -> device upload
            bind[self.const_names[j]] = self.dtensors[j]
        self.req.set_tensors(bind)
        return (time.perf_counter() - t0) * 1e3

    def step(self, lat: np.ndarray, tstep: np.ndarray) -> np.ndarray:
        """One Euler step: re-bind only the two changing inputs, infer, copy the output out."""
        import openvino as ov

        self.req.set_tensors(
            {self.host_names[0]: ov.Tensor(as_input(self.host_prec[0], lat)), self.host_names[1]: ov.Tensor(as_input(self.host_prec[1], tstep))}
        )
        self.req.infer()
        return self.req.get_output_tensor(0).data.copy()  # copy: the next infer overwrites it


class PrefillAsync:
    """The video prefill on its own InferRequest, so host-side context work can hide under it (``overlap``).

    Same graph, same bytes, same inputs; only the *order* of the host-side work changes: issue the
    prefill with ``start_async``, do the context cast / device uploads while it is in flight, then
    ``wait()``.  The 16 K/V outputs cannot be hidden (they are the prefill result), so the hidden
    part is reported separately.
    """

    def __init__(self, unit: Unit) -> None:
        self.unit = unit
        self.req = unit.compiled.create_infer_request()
        self.keep = None

    def start(self, *tensors) -> None:
        """Cast the 4 graph inputs exactly like ``Unit.__call__`` and go async."""
        import openvino as ov

        cols = [as_input(p, np.asarray(t)) for p, t in zip(self.unit.in_prec, tensors)]
        self.keep = [ov.Tensor(a) for a in cols]  # refs must outlive the request
        self.req.set_tensors({n: t for n, t in zip(self.unit.in_names, self.keep)})
        self.req.start_async()

    def wait(self) -> list[np.ndarray]:
        """Block, then hand back the outputs exactly like ``Unit.__call__`` does."""
        self.req.wait()
        return [np.asarray(self.req.get_output_tensor(j).data) for j in range(len(self.unit.compiled.outputs))]


def load_units(paths: dict[str, Path], device: str, threads: int | None = None, log: bool = True, kvdev: bool = True, overlap: bool = True) -> dict[str, Unit]:
    """Compile the three units of one IR set onto one device.

    Args:
        paths: output of :func:`ir_paths`.
        device: target device, for example ``GPU``.
        threads: CPU inference threads (only used when ``device`` starts with ``CPU``); ``None``
            (default) leaves the choice to the CPU plugin.
        log: print a compile line per unit.
        kvdev: keep the constant-per-chunk action inputs resident in DEVICE memory
            (:class:`DeviceConstInputs`).  Needs f16 action-graph inputs, i.e. the INT8 set; the fp16
            set (f32 graph inputs) and any device without a remote context take the host path, with
            one printed line.
        overlap: issue the video prefill asynchronously so host-side context work hides under it
            (:class:`PrefillAsync`).  On for a GPU, ignored for the CPU.

    Returns:
        ``{"vae": Unit, "video": Unit, "action": Unit}``.
    """
    import openvino as ov

    core = ov.Core()
    units = {}
    is_gpu = device.upper().startswith("GPU")
    for name in UNITS:
        units[name] = Unit(core, paths[name], device, threads=threads)
        if log:
            print(f"[ov] {name:6s} {device:6s} weights={units[name].weight_mb:7.1f} MB compile={units[name].compile_s:5.1f} s")
    if kvdev and is_gpu:
        try:
            units["action"]._dev_inputs = DeviceConstInputs(units["action"])
            di = units["action"]._dev_inputs
            if log:
                print(
                    f"[ov] kvdev: {len(di.dev_idx)} constant inputs resident on {device}, "
                    f"{di.nbytes / 1e6:.1f} MB uploaded once per chunk (setup {di.setup_s} s)"
                )
        except Exception as exc:  # noqa: BLE001
            units["action"]._dev_inputs = None
            if log:
                print(f"[ov] kvdev: not used, K/V stay host inputs ({exc})")
    if overlap and is_gpu:
        units["video"]._async_prefill = PrefillAsync(units["video"])
        if log:
            print("[ov] overlap: video prefill on a dedicated InferRequest (same compiled graph)")
    return units


def build_schedule(steps: int, shift: float = 5.0, num_train_timesteps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Return the flow-matching Euler schedule used by the action loop.

    Args:
        steps: number of Euler steps (10 in the released checkpoint).
        shift: scheduler shift (5.0 for the released checkpoint).
        num_train_timesteps: 1000 for the released checkpoint.

    Returns:
        ``(t_sched, d_sched)``, each of length ``steps``: the timestep fed to the graph and the
        multiplier applied to its prediction.
    """
    u = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)
    sigma = shift * u / (1.0 + (shift - 1.0) * u)
    return sigma[:-1] * float(num_train_timesteps), sigma[1:] - sigma[:-1]


def run_chunk(units: dict[str, Unit], inp: dict, steps: int = 10, shift: float = 5.0, seed: int = 42) -> dict:
    """Run one full action chunk: VAE encode, video prefill, then ``steps`` action steps.

    Args:
        units: output of :func:`load_units`.
        inp: graph inputs; see :func:`sample_inputs` and :func:`eval_inputs`.
        steps: Euler steps for the action loop.
        shift: flow-matching scheduler shift.
        seed: seed for the numpy noise, used for whichever of ``lat_video`` / ``lat_action`` ``inp``
            does not carry (video is drawn first, then action, from one generator).

    Returns:
        ``{"action": [32, 7] float32, "timing": {...ms}, "first_frame": latent, "kv": [...]}``
        - the chunk horizon is fixed by the graph, ``steps`` only sets the Euler steps - and ``kv`` holds the 16 K/V tensors exactly
        as they were bound to the action graph - the list is handed back as it is, so returning it
        costs nothing extra (the cast that produced it is the measured ``kv_cast``).

    Note:
        ``kv`` can be a numpy **view of the video request's output buffers** (it is on the GPU, where
        the async prefill hands its outputs over without a copy): the next prefill overwrites it.
        Reading it right after the call is fine; keeping it past another ``run_chunk`` (or a
        benchmark) requires ``np.array(...)`` first.
    """
    vae, video, action = units["vae"], units["video"], units["action"]
    timing = {}

    t0 = time.perf_counter()
    image = np.asarray(inp["image"], dtype=np.float32)
    if image.ndim == 4 and len(vae.in_shapes[0]) == 5:  # the VAE graph is video-shaped: [1,3,T,H,W]
        image = np.expand_dims(image, 2)
    first_frame = vae(image)[0]
    timing["vae_ms"] = (time.perf_counter() - t0) * 1e3

    rng = None if ("lat_video" in inp and "lat_action" in inp) else np.random.default_rng(seed)
    if "lat_video" in inp:
        lat_video = np.array(inp["lat_video"], dtype=np.float32)
    else:
        lat_video = rng.standard_normal(video.in_shapes[0]).astype(np.float32)
    if "lat_action" in inp:
        lat_action = np.array(inp["lat_action"], dtype=np.float32)
    else:
        lat_action = rng.standard_normal(action.in_shapes[0]).astype(np.float32)
    lat_video[:, :, 0:1] = first_frame.astype(np.float32)

    t_sched, d_sched = build_schedule(steps, shift)
    di = getattr(action, "_dev_inputs", None)  # kvdev: device-resident constant inputs
    asyn = getattr(video, "_async_prefill", None)  # overlap: async video prefill
    pf_args = (lat_video, np.array([t_sched[0]], np.float32), inp["context"], inp["context_mask"])

    t0 = time.perf_counter()
    if asyn is not None:
        asyn.start(*pf_args)  # the prefill goes in flight
        timing["prefill_issue_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        t1 = time.perf_counter()  # ... hide the ctx-side work under it
        ctx = as_input(action.in_prec[2], inp["context"])
        if di is not None:
            timing["kv_upload_ovl_ms"] = round(di.upload_part([(0, ctx)]), 2)
        timing["overlap_ms"] = round((time.perf_counter() - t1) * 1e3, 2)
        t2 = time.perf_counter()
        kvs = asyn.wait()
        timing["prefill_wait_ms"] = round((time.perf_counter() - t2) * 1e3, 2)
        timing["prefill_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        if len(kvs) != 2 * len(COND_LAYERS):
            raise RuntimeError(f"expected {2 * len(COND_LAYERS)} video K/V outputs, got {len(kvs)}")
        t3 = time.perf_counter()
        kv = [as_input(prec, k) for prec, k in zip(action.in_prec[3:], kvs)]
        if di is not None:
            timing["kv_upload_ms"] = round(di.upload_part([(1 + j, a) for j, a in enumerate(kv)]), 2)
        timing["kv_cast_ms"] = round((time.perf_counter() - t3) * 1e3, 2)
    else:
        kvs = video(*pf_args)
        timing["prefill_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        if len(kvs) != 2 * len(COND_LAYERS):
            raise RuntimeError(f"expected {2 * len(COND_LAYERS)} video K/V outputs, got {len(kvs)}")
        t1 = time.perf_counter()
        ctx = as_input(action.in_prec[2], inp["context"])
        kv = [as_input(prec, k) for prec, k in zip(action.in_prec[3:], kvs)]
        if di is not None:
            timing["kv_upload_ms"] = round(di.upload([ctx, *kv]), 2)
        timing["kv_cast_ms"] = round((time.perf_counter() - t1) * 1e3, 2)

    t0 = time.perf_counter()
    for i in range(steps):
        tstep = np.array([t_sched[i]], np.float32)
        if di is not None:
            pred = di.step(lat_action, tstep)  # constant inputs never leave the device
        else:
            pred = action(as_input(action.in_prec[0], lat_action), tstep, ctx, *kv)[0]
        lat_action = lat_action + pred.astype(np.float32) * np.float32(d_sched[i])
    timing["action_loop_ms"] = (time.perf_counter() - t0) * 1e3
    timing["action_step_ms"] = timing["action_loop_ms"] / steps

    # wall-clock decomposition: prefill_ms already spans issue+overlap+wait and kv_upload_ms sits
    # inside kv_cast_ms, so neither is added twice (they are reported for insight only)
    timing["total_ms"] = timing["vae_ms"] + timing["prefill_ms"] + timing["kv_cast_ms"] + timing["action_loop_ms"]
    for key in ("vae_ms", "prefill_ms", "kv_cast_ms", "action_loop_ms", "action_step_ms", "total_ms"):
        timing[key] = round(timing[key], 2)
    actions = lat_action[0] if lat_action.ndim == 3 else lat_action
    return {"action": np.asarray(actions, dtype=np.float32), "timing": timing, "first_frame": first_frame, "kv": kv}


def bench(
    units: dict[str, Unit], inp: dict, steps: int = 10, niter: int = 5, warmup: int = 2, seed: int = 42, text_encoder=None, prompt: str | None = None
) -> dict:
    """Time ``niter`` chunks after ``warmup`` untimed ones, optionally including the text encode.

    The warmup is not optional for an INT8 set: its compressed-weight kernels are initialized
    lazily on the first inference of each graph and cost about 60 ms extra for the video prefill.
    It is just as necessary for a ``text_encoder``, whose first inference after ``compile_model``
    pays a one-time setup of a few hundred milliseconds.

    Args:
        units: output of :func:`load_units`.
        inp: graph inputs.
        steps: Euler steps.
        niter: timed chunks.
        warmup: untimed chunks first.
        seed: noise seed.
        text_encoder: a :class:`TextEncoder`.  When given, every iteration encodes ``prompt`` and
            feeds the fresh context through :func:`with_text_context`, so what is measured starts
            at the instruction instead of at a cached context array.
        prompt: the instruction to encode (falls back to ``inp["prompt"]`` if the dict has one);
            required with a ``text_encoder``.

    Returns:
        ``{stage: {"mean_ms", "p50_ms", "min_ms"}}``, always including ``total_ms`` (the chunk) and
        - when a ``text_encoder`` was passed - ``t5_ms`` (tokenize + encode) and ``end_to_end_ms``
        (that iteration's ``t5_ms`` plus its ``total_ms``).  Every other stage is untouched, so a
        caller that passes no encoder gets exactly the numbers it got before.
    """
    if text_encoder is not None:
        if prompt is None:
            prompt = inp.get("prompt")
        if prompt is None:
            raise ValueError("bench(text_encoder=...) needs an instruction: pass prompt=...")

    def one() -> dict:
        """One chunk, preceded by one instruction encode when an encoder was given."""
        text_ms = None
        chunk_inp = inp
        if text_encoder is not None:
            t0 = time.perf_counter()
            context, _, _ = text_encoder.encode([prompt])
            text_ms = (time.perf_counter() - t0) * 1e3
            chunk_inp = with_text_context(inp, context)
        timing = run_chunk(units, chunk_inp, steps=steps, seed=seed)["timing"]
        if text_ms is not None:
            timing["t5_ms"] = round(text_ms, 2)
            timing["end_to_end_ms"] = round(text_ms + timing["total_ms"], 2)
        return timing

    for _ in range(warmup):
        one()
    rows = [one() for _ in range(niter)]
    out = {}
    for key in rows[0]:
        values = np.array([r[key] for r in rows], dtype=np.float64)
        out[key] = {"mean_ms": round(float(values.mean()), 2), "p50_ms": round(float(np.median(values)), 2), "min_ms": round(float(values.min()), 2)}
    return out


# --------------------------------------------------------------------------- inputs and metrics


def sample_inputs(pack_path: str | Path, index: int = 0) -> dict:
    """Load the self-contained regression pack: graph inputs, noise, and the torch reference.

    Args:
        pack_path: path to ``ov_runtime_pack.npz``.
        index: ignored; kept for compatibility.  The released pack holds exactly one sample and its
            arrays carry no sample axis, so there is nothing to index.

    Returns:
        A dict ready for :func:`run_chunk`, plus ``"action_ref"`` (the fp32 torch result for the same
        inputs).  The noise (``lat_video``/``lat_action``) comes from the pack, so a chunk is
        reproducible without drawing any.
    """
    with np.load(pack_path) as z:
        return {
            "image": z["image"].astype(np.float32),
            "lat_video": np.array(z["lat_video"]),
            "lat_action": np.array(z["lat_action"]),
            "context": np.array(z["ctx_p"]),
            "context_mask": np.array(z["ctx_pm"]),
            "action_ref": z["action_ref"].astype(np.float32),
        }


def eval_tasks(art_dir: str | Path) -> list[str]:
    """Return the language instructions of the evaluation set, one per sample.

    Args:
        art_dir: directory holding the ``eval_*.npz`` artifacts and their JSON sidecar.

    Returns:
        The task strings, aligned with the sample index used by :func:`eval_inputs`.
    """
    sidecar = json.loads((Path(art_dir) / "eval_samples.json").read_text())
    return [sample["task"] for sample in sidecar["samples"]]


def eval_inputs(art_dir: str | Path, index: int = 0) -> dict:
    """Load one sample of the 32-sample LIBERO evaluation set.

    The artifacts are aligned by index: ``eval_samples.npz`` holds the dual-camera mosaic, the
    normalised proprio state and the ground-truth action chunk, ``eval_contexts.npz`` holds the
    umT5 text context of the same instruction, and ``eval_latents.npz`` the noise that makes a
    torch-free host reproduce the recorded reference run.

    Args:
        art_dir: directory holding the ``eval_*.npz`` artifacts.
        index: sample index, 0..31.

    Returns:
        A dict ready for :func:`run_chunk` plus ``"gt_action"`` (normalised ground truth) and
        ``"image_hwc"`` (the uint8 observation mosaic).
    """
    art = Path(art_dir)
    with np.load(art / "eval_samples.npz") as z, np.load(art / "eval_contexts.npz") as c, np.load(art / "eval_latents.npz") as lat, np.load(
        art / "eval_proprio.npz"
    ) as pe:
        # The 128 text tokens are followed by one proprioception token: state @ W.T + b, projected
        # into the same 4096-wide space, so the stream the graphs see is 129 tokens long.
        token = (z["state_norm"][index : index + 1].astype(np.float32) @ pe["weight"].T + pe["bias"])[:, None, :]
        return {
            "image": np.transpose(z["image_norm"][index : index + 1], (0, 3, 1, 2)).astype(np.float32),
            "lat_video": np.array(lat["lat_video"][0]),
            "lat_action": np.array(lat["lat_action"][0]),
            "context": np.concatenate([c["context"][index : index + 1], token], axis=1).astype(np.float32),
            "context_mask": np.concatenate([c["mask"][index : index + 1], np.ones((1, 1), np.float32)], axis=1),
            "gt_action": z["gt_action_norm"][index].astype(np.float32),
            "image_hwc": np.array(z["image"][index]),
        }


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Compare two action chunks.

    Args:
        reference: reference actions, for example the torch result.
        candidate: actions to judge.

    Returns:
        ``{"cos", "max_abs", "rmse"}`` over the flattened chunk.  ``cos`` is rounded to 8 decimals so a
        reported number reads the same everywhere; ``rmse`` and ``max_abs`` are left exact, because
        small errors are the interesting ones in the accuracy section.  ``cos`` is the headline
        number used throughout the FasterWAM project (0.9999 and above for a good INT8 set).
    """
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(candidate, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "cos": round(float(a @ b) / denom, 8) if denom else float("nan"),
        "max_abs": float(np.max(np.abs(a - b))),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
    }
