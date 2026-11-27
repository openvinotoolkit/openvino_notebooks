#!/usr/bin/env python
"""PyTorch-on-the-CPU reference action chunk for the FasterWAM OpenVINO notebook.

The notebook never imports torch - the export runs in this kind of subprocess and inference needs
nothing but OpenVINO.  The accuracy section needs the other side of the comparison, so this script
does the opposite of the exporter: it builds the model from the source clone plus the released
checkpoint (the exporter's own recipe, ``utils/fwam_export/run_infer_smoke.py``), replays exactly
what ``fwam_ov.run_chunk`` asks the three OpenVINO graphs to do, and writes the numbers to an
``.npz`` so the notebook can compare against them with numpy only::

    vae      image                     -> first latent frame
    video    latents[1,48,3,14,28] ... -> the 16 fused K/V tensors of the 8 tapped blocks
    action   latents[1,32,7] + t, x10  -> pred[1,32,7], Euler-updated

Everything runs eagerly on the CPU in one dtype (``--dtype``), no CUDA, no attention kernels.

Usage (from the notebook folder)::

    .venv/bin/python torch_ref_actions.py                      # dtype auto, 10 steps, all cores
    .venv/bin/python torch_ref_actions.py --dtype float32      # needs ~25 GB of RAM for weights

``--dtype auto`` (the default) runs float32 when the host has about 40 GiB available and the
checkpoint's own bfloat16 otherwise.

Cost on a 16-core laptop: about 30-40 s to instantiate and load the checkpoint, then about 3-4 s
for the chunk.  The result is cached in ``artifacts/`` - delete the ``.npz`` to recompute it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import resource  # not on Windows; only used for the peak-RSS report
except ImportError:  # pragma: no cover
    resource = None

HERE = Path(__file__).resolve().parent
EXPORT_DIR = HERE / "utils" / "fwam_export"
# fwam_ov sits next to this script and imports only numpy and the standard library, so the shared
# constants (checkpoint location, tapped blocks, the fp32 RAM rule, the cache identity) come from it.
sys.path.insert(0, str(HERE))
import fwam_ov  # noqa: E402

COND_LAYERS = fwam_ov.COND_LAYERS  # the blocks whose K/V the video graph taps
available_gib = fwam_ov.available_gib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default=str(HERE), help="the notebook folder (it owns model/)")
    parser.add_argument("--pack", default=None, help="input pack (default <workdir>/artifacts/ov_runtime_pack.npz)")
    parser.add_argument("--out", default=None, help="output .npz (default <workdir>/artifacts/torch_ref_pack.npz)")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "float32"),
        help="model dtype. 'auto' picks float32 when the host has the RAM for it and "
        "bfloat16 otherwise (fp32 weights alone are ~25 GB; bfloat16 is what the "
        "released checkpoint stores)",
    )
    parser.add_argument("--steps", type=int, default=10, help="Euler steps (the checkpoint's 10)")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    return parser.parse_args()


def resolve_dtype(name: str) -> tuple[str, str]:
    """Return ``(dtype name, why)``; ``auto`` asks for fp32 only when the host can hold it."""
    if name != "auto":
        return name, "requested"
    available = available_gib()
    # fp32 needs ~25 GB of weights plus the page cache of the 11 GB checkpoint it maps; below that
    # the run does not fail, it swaps, which on this class of machine is minutes of random I/O.
    if available >= fwam_ov.TORCH_REF_FP32_GIB:
        return "float32", f"{available:.0f} GiB available, so the exact fp32 model fits"
    return "bfloat16", (f"only {available:.0f} GiB available and an fp32 model needs ~37 GiB; " f"bfloat16 is the checkpoint's own storage dtype")


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two flattened arrays."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error of two arrays of the same shape."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.sqrt((d * d).mean()))


def peak_rss_gib() -> float:
    """Return this process's peak RSS in GiB (``ru_maxrss`` is KiB on Linux, bytes on macOS)."""
    if resource is None:
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


def mmap_torch_load(torch) -> None:
    """Make the repo's ``torch.load(path, map_location="cpu")`` map the checkpoint instead of reading it.

    The released checkpoint is 11 GB; mapping it keeps that 11 GB out of RSS, which is the difference
    between this script fitting next to a compiled OpenVINO pipeline and not.
    """
    original = torch.load

    def load(*args, **kwargs):  # noqa: ANN002,ANN003
        kwargs.setdefault("map_location", "cpu")
        try:
            return original(*args, **kwargs, mmap=True)
        except (RuntimeError, ValueError, TypeError) as exc:  # pragma: no cover - old/pickle format
            print(f"[torch] mmap load unavailable ({exc}); reading it into RAM", flush=True)
            return original(*args, **kwargs)

    torch.load = load


def build_model(torch, clone: Path, ckpt: Path, dtype, steps: int):
    """Instantiate FasterWAM on the CPU and load the released checkpoint (the exporter's recipe)."""
    import run_infer_smoke as RIS  # noqa: E402  (needs REPO/PYTHONPATH, set in main)

    t0 = time.perf_counter()
    cfg = RIS.load_cfg("fasterwam", None)  # the LIBERO task config the notebook exports
    _ns, spec = RIS.make_spec("fasterwam", cfg, None, action_horizon=None, num_inference_steps=steps, synthetic_seed=0)
    nargs = SimpleNamespace(ckpt=str(ckpt), load_text_encoder=False)
    model, facts = RIS.build_ckpt(nargs, cfg, dtype)
    model.eval()
    torch.set_grad_enabled(False)
    return model, spec, facts, round(time.perf_counter() - t0, 1)


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    clone = workdir / "model" / "FasterWAM"
    ckpt = fwam_ov.checkpoint_path(workdir)
    pack_path = Path(args.pack) if args.pack else workdir / "artifacts" / "ov_runtime_pack.npz"
    out_path = Path(args.out) if args.out else workdir / "artifacts" / "torch_ref_pack.npz"
    for path, what in ((clone / "src" / "fasterwam", "source clone"), (ckpt, "checkpoint"), (pack_path, "input pack")):
        if not path.exists():
            raise SystemExit(f"[torch] missing {what}: {path}")

    # The repo harness reads these at import time; the exporter sets the same three.
    os.environ["REPO"] = str(clone)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(clone / "checkpoints")
    os.environ["PYTHONPATH"] = str(clone / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    threads = args.threads or (os.cpu_count() or 8)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    sys.path.insert(0, str(clone / "src"))
    sys.path.insert(0, str(clone / "experiments" / "latency"))
    sys.path.insert(0, str(EXPORT_DIR))
    os.chdir(clone)  # the config loader and the hydra compose below are relative to the clone

    import torch

    torch.set_num_threads(threads)
    mmap_torch_load(torch)
    dtype_name, why = resolve_dtype(args.dtype)
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16

    with np.load(pack_path) as pack:
        z = {k: np.array(pack[k]) for k in pack.files}
    identity = fwam_ov.torch_ref_identity(pack_path, ckpt)  # what this reference is computed from
    print(f"[torch] {torch.__version__} on the CPU, {threads} threads, dtype {dtype_name} ({why})", flush=True)
    print(f"[torch] building the model from {clone} ...", flush=True)
    model, spec, facts, build_s = build_model(torch, clone, ckpt, dtype, args.steps)
    print(f"[torch] built in {build_s} s (missing/unexpected checkpoint keys: " f"{facts['n_missing']}/{facts['n_unexpected']})", flush=True)
    dev = torch.device("cpu")
    to_dev = lambda a, dt=dtype: torch.from_numpy(np.ascontiguousarray(a)).to(device=dev, dtype=dt)

    # The pack's context is already the 129-token stream the graphs see (128 text tokens plus the
    # projected proprioception token), so `_append_proprio_to_context` is not part of the replay.
    image = to_dev(z["image"])
    context = to_dev(z["ctx_p"], dtype)
    context_mask = torch.from_numpy(np.ascontiguousarray(z["ctx_pm"])).to(device=dev, dtype=torch.bool)
    lat_video = to_dev(z["lat_video"], dtype)
    lat_action = to_dev(z["lat_action"], dtype)
    timing: dict[str, float] = {}

    with torch.inference_mode():
        # 1. VAE: the observation frame -> the first latent frame of the video latent.
        t0 = time.perf_counter()
        first_frame = model._encode_input_image_latents_tensor(input_image=image, tiled=False)
        timing["vae_s"] = time.perf_counter() - t0
        lat_video[:, :, 0:1] = first_frame

        # 2. Video prefill: one forward pass of the 5 B video expert, which is where the 16 K/V
        #    tensors the action expert conditions on come from.
        t0 = time.perf_counter()
        fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
        t_video, _ = model.infer_video_scheduler.build_inference_schedule(num_inference_steps=args.steps, device=dev, dtype=dtype)
        video_pre = model.video_expert.pre_dit(
            x=lat_video,
            timestep=t_video[0].unsqueeze(0).to(dtype=dtype, device=dev),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        per_frame = int(video_pre["meta"]["tokens_per_frame"])
        attention_masks = model._build_mot_attention_masks(
            video_seq_len=video_seq_len,
            action_seq_len=lat_action.shape[1],
            video_tokens_per_frame=per_frame,
            device=dev,
        )
        video_attention_mask = model._build_video_attention_mask(video_seq_len=video_seq_len, video_tokens_per_frame=per_frame, device=dev)
        kv_cache = model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=video_attention_mask,
        )
        timing["prefill_s"] = time.perf_counter() - t0
        print(f"[torch] video prefill {timing['prefill_s']:.2f} s " f"({video_seq_len} tokens, {per_frame} per frame)", flush=True)

        # 3. Action loop: the same ten Euler steps, reusing that K/V cache.
        t0 = time.perf_counter()
        t_action, d_action = model.infer_action_scheduler.build_inference_schedule(num_inference_steps=args.steps, device=dev, dtype=dtype)
        for step, (t_step, d_step) in enumerate(zip(t_action, d_action)):
            pred = model._predict_action_noise_with_cache(
                latents_action=lat_action,
                timestep_action=t_step.unsqueeze(0).to(dtype=dtype, device=dev),
                context=context,
                context_mask=context_mask,
                video_kv_cache=kv_cache,
                attention_mask=attention_masks,
                video_seq_len=video_seq_len,
            )
            lat_action = model.infer_action_scheduler.step(pred, d_step, lat_action)
            if step == 0:
                print(f"[torch] action step 1/{args.steps} took {(time.perf_counter() - t0) * 1e3:.0f} ms", flush=True)
        timing["action_loop_s"] = time.perf_counter() - t0
        action = lat_action[0].detach().to(dtype=torch.float32).numpy()

        kv = np.zeros((2 * len(COND_LAYERS),) + tuple(kv_cache[COND_LAYERS[0]]["k"].shape), np.float32)
        for layer, index in enumerate(COND_LAYERS):
            kv[2 * layer] = kv_cache[index]["k"].detach().float().numpy()
            kv[2 * layer + 1] = kv_cache[index]["v"].detach().float().numpy()

    timing["total_s"] = timing["vae_s"] + timing["prefill_s"] + timing["action_loop_s"]
    timing["step_ms"] = timing["action_loop_s"] * 1e3 / args.steps
    timing["peak_rss_gib"] = peak_rss_gib()

    # Self-checks the notebook prints: the pack ships a torch *fp32* reference for this same chunk,
    # and the fp16 video K/V outputs, so a wrong replay cannot pass unnoticed.
    # The pack may legitimately carry no reference: `make_eval_packs.py` stages one to *bootstrap*
    # this script (there is nothing to compare against on the very first run).  Compare when it is
    # there, and say so when it is not.
    reference_keys = ["action_ref", "first_frame"] + [f"kv{i:02d}" for i in range(2 * len(COND_LAYERS))]
    have_reference = all(key in z for key in reference_keys)
    if have_reference:
        action_ref = z["action_ref"].astype(np.float32)
        first_frame_ref = z["first_frame"].astype(np.float32)
        kv_ref = np.stack([z[f"kv{i:02d}"].astype(np.float32) for i in range(2 * len(COND_LAYERS))])
        checks = {
            "action_vs_pack_cos": cos(action_ref, action),
            "action_vs_pack_rmse": rmse(action_ref, action),
            "first_frame_vs_pack_cos": cos(first_frame_ref, first_frame.float().numpy()),
            "kv_vs_pack_cos": cos(kv_ref, kv),
            "kv_vs_pack_rmse": rmse(kv_ref, kv),
        }
    else:
        checks = {}
        print(
            "[torch] the pack carries no torch reference - this is a bootstrap run, so the " "self-checks are skipped and the result becomes the reference",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        action=action.astype(np.float32),
        first_frame=first_frame.float().numpy().astype(np.float32),
        kv=kv,
        dtype=np.array(dtype_name),
        steps=np.array(args.steps),
        threads=np.array(threads),
        build_s=np.array(build_s),
        timing=np.array(json.dumps({k: round(v, 4) for k, v in timing.items()})),
        checks=np.array(json.dumps({k: round(v, 9) for k, v in checks.items()})),
        identity=np.array(json.dumps(identity)),
    )
    print(
        f"[torch] chunk {timing['total_s']:.2f} s "
        f"(vae {timing['vae_s']:.2f} + prefill {timing['prefill_s']:.2f} + "
        f"{args.steps} steps {timing['action_loop_s']:.2f}), "
        f"peak RSS {timing['peak_rss_gib']:.1f} GiB",
        flush=True,
    )
    if checks:
        print(
            f"[torch] self-check vs the pack's torch fp32 reference: "
            f"action cos {checks['action_vs_pack_cos']:.8f}, "
            f"first frame cos {checks['first_frame_vs_pack_cos']:.8f}, "
            f"K/V cos {checks['kv_vs_pack_cos']:.8f}",
            flush=True,
        )
    print(f"[torch] wrote {out_path}", flush=True)
    print("SUMMARY " + json.dumps({"out": str(out_path), "dtype": dtype_name, "steps": args.steps, "build_s": build_s, **timing, **checks}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
