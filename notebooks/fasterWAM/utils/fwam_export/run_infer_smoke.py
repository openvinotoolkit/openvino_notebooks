#!/usr/bin/env python
"""CPU action-inference smoke test / baseline for FasterWAM.

Reuses the repo's own latency harness (`experiments/latency/measure_latency.py`) for
config + architecture + input construction, so the architecture under test is exactly the
one the paper's benchmark times.  Only CPU-safe parts of that module are used
(`load_config`, `make_run_spec`, `build_model`, `prepare_inputs`); its `main()`,
`cuda_sync()` and `profile_cached()` are CUDA-only and are never called.

The repo is READ-ONLY from our point of view: everything adapted here lives in this file.

Usage (see run_infer_smoke.sh):
    scripts/run_infer_smoke.sh --weights random --dtype float32 --steps 1 --iters 1
    scripts/run_infer_smoke.sh --weights ckpt --ckpt $REPO/checkpoints/hustvl/FasterWAM/libero/step_021700.pt \
        --steps 1 --iters 1 --dump scripts/artifacts/smoke_real_ckpt.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "24")
os.environ.setdefault("MKL_NUM_THREADS", "24")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(os.environ.get("REPO", "/home/clee30/disk2/models/fasterWAM/FasterWAM")).resolve()
SCRIPTS = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "experiments" / "latency"))

import measure_latency as ml  # noqa: E402  (repo latency harness, CPU-safe functions only)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FasterWAM CPU action-inference smoke test.")
    p.add_argument("--model", default="fasterwam", choices=["fasterwam", "fastwam", "jointwam"])
    p.add_argument(
        "--task", default=None, help="hydra task override (default: measure_latency's model->LIBERO mapping). " "e.g. robotwin_fasterwam_3cam_384_1e-4"
    )
    p.add_argument("--weights", default="random", choices=["random", "ckpt"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--steps", type=int, default=1, help="num_inference_steps for the action loop")
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--action-horizon", dest="action_horizon", type=int, default=None)
    p.add_argument("--synthetic-seed", dest="synthetic_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=42, help="latent noise seed passed to infer_*")
    p.add_argument("--prompt", default=None, help="text prompt (needs --load-text-encoder)")
    p.add_argument(
        "--load-text-encoder",
        action="store_true",
        help="ckpt mode: also load the 11 GB umT5 encoder + tokenizer (needed for --prompt)",
    )
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--dump", default=None, help="torch.save inputs + output action to this path")
    p.add_argument(
        "--steps-sweep",
        default=None,
        help="comma list, e.g. 1,3,10,20: time infer() once per value in ONE process " "(single model build) to get a clean per-action-step slope",
    )
    p.add_argument(
        "--stage-timing",
        action="store_true",
        help="time the individual units (VAE enc / masks / video prefill / 1 action step) " "by replaying infer_action_one_pass_future_cache's internals",
    )
    p.add_argument("--build-model-time", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.bfloat16


def load_cfg(model: str = "fasterwam", task: str | None = None):
    """`ml.load_config` with an optional task override.

    The repo helper hardcodes `task=TASK_BY_MODEL[model]`, i.e. the LIBERO task, so the released
    `configs/task/robotwin_*.yaml` (3 cams, video_size 384x320, action/proprio dim 14, z-score
    stats) is unreachable from it.  Same compose call, same CONFIG_DIR, repo stays read-only.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    name = task or ml.TASK_BY_MODEL[model]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ml.CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=[f"task={name}"])
    return cfg


def prepare_inputs(spec, device) -> dict:
    """ml.prepare_inputs, but with `proprio` matching spec.proprio_dim.

    The repo helper builds one fixed 8-value vector and *slices* it to proprio_dim, so a 14-dim
    task (robotwin) would silently get [1,8] and die in _append_proprio_to_context.  For
    proprio_dim <= 8 the tensors are byte-identical to the repo's.
    """
    inputs = ml.prepare_inputs(spec, device)
    want = int(spec.proprio_dim)
    if int(inputs["proprio"].shape[-1]) != want:
        base = torch.tensor([[0.45, 0.0, 0.25, 0.0, 0.0, 0.0, 0.04, 0.04]], dtype=torch.float32)
        inputs["proprio"] = base.repeat(1, int(np.ceil(want / 8)))[:, :want].to(device)
    return inputs


def make_spec(model: str, cfg, task: str | None = None, **overrides):
    """ml.make_run_spec + the correct `task` label (RunSpec is frozen -> dataclasses.replace)."""
    import dataclasses

    ns = SimpleNamespace(
        **{
            "action_horizon": overrides.get("action_horizon"),
            "num_inference_steps": overrides.get("num_inference_steps"),
            "synthetic_seed": overrides.get("synthetic_seed", 0),
        }
    )
    spec = ml.make_run_spec(model, cfg, ns)
    if task and spec.task != task:
        spec = dataclasses.replace(spec, task=task)
    return ns, spec


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def count_params(model: torch.nn.Module) -> dict[str, int]:
    out = {}
    for name in ("video_expert", "action_expert", "mot", "vae", "text_encoder", "proprio_encoder"):
        sub = getattr(model, name, None)
        if sub is None:
            continue
        out[name] = sum(int(p.numel()) for p in sub.parameters())
    out["total"] = sum(int(p.numel()) for p in model.parameters())
    return out


def build_random(args: argparse.Namespace, cfg: Any, dtype: torch.dtype):
    device = torch.device("cpu")
    start = time.perf_counter()
    model = ml.build_model(args.model, cfg, device)
    build_s = time.perf_counter() - start
    if dtype is torch.float32:
        model.float()
    else:
        model.bfloat16()
    model.torch_dtype = dtype
    return model, build_s


def build_ckpt(args: argparse.Namespace, cfg: Any, dtype: torch.dtype):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if args.ckpt is None:
        raise SystemExit("--weights ckpt requires --ckpt PATH")
    # Mirror experiments/libero/eval_libero_single.py:720-728 (instantiate + load_checkpoint).
    # This OmegaConf build has no `read_write` context manager -> clear struct/readonly flags
    # on every node we touch.
    for node in (cfg, cfg.model, cfg.model.get("video_dit_config"), cfg.model.get("action_dit_config")):
        if node is not None:
            OmegaConf.set_struct(node, False)
            OmegaConf.set_readonly(node, False)
    cfg.model.load_text_encoder = bool(args.load_text_encoder)
    cfg.model.skip_dit_load_from_pretrain = True  # ckpt is the sole source of DiT weights
    cfg.model.action_dit_pretrained_path = None
    cfg.model.redirect_common_files = False  # shipped .pth files match the md5 registry
    cfg.model.mot_checkpoint_mixed_attn = False  # inference never checkpoints
    for sub in ("video_dit_config", "action_dit_config"):
        if sub in cfg.model:
            cfg.model[sub]["use_gradient_checkpointing"] = False

    start = time.perf_counter()
    model = instantiate(cfg.model, model_dtype=dtype, device="cpu")
    instantiate_s = time.perf_counter() - start

    # Capture the strict=False missing/unexpected keys without editing the repo: wrap
    # mot.load_state_dict just around FastWAM.load_checkpoint's call.
    captured: dict[str, list[str]] = {}
    mot = model.mot
    original = mot.load_state_dict

    def spy(state_dict, strict=True):  # noqa: ANN001
        result = original(state_dict, strict=strict)
        captured["missing"] = list(result.missing_keys)
        captured["unexpected"] = list(result.unexpected_keys)
        return result

    mot.load_state_dict = spy
    start = time.perf_counter()
    payload = model.load_checkpoint(str(args.ckpt))
    load_s = time.perf_counter() - start
    mot.load_state_dict = original

    model = model.to("cpu").eval()
    facts = {
        "instantiate_seconds": instantiate_s,
        "load_checkpoint_seconds": load_s,
        "payload_keys": sorted(str(k) for k in payload.keys()) if isinstance(payload, dict) else None,
        "payload_step": payload.get("step") if isinstance(payload, dict) else None,
        "payload_torch_dtype": payload.get("torch_dtype") if isinstance(payload, dict) else None,
        "missing_keys": captured.get("missing"),
        "unexpected_keys": captured.get("unexpected"),
        "n_missing": len(captured.get("missing", [])),
        "n_unexpected": len(captured.get("unexpected", [])),
        "missing_sample": captured.get("missing", [])[:20],
        "unexpected_sample": captured.get("unexpected", [])[:20],
        "model_paths": getattr(model, "model_paths", None),
    }
    if isinstance(payload, dict):
        for k in ("optimizer",):
            payload.pop(k, None)
    del payload
    gc.collect()
    return model, facts


def stage_timing(model, spec, inputs, steps: int, reps: int = 2) -> dict[str, Any]:
    """Replay the internals of infer_action_one_pass_future_cache (jointwam.py:264-424)
    stage by stage so each export unit gets its own CPU cost. Nothing here edits the repo."""
    out: dict[str, Any] = {}
    dev = torch.device("cpu")
    dt = model.torch_dtype
    img = inputs["input_image"]

    def timeit(fn, n=reps):
        best = None
        for _ in range(n):
            t0 = time.perf_counter()
            res = fn()
            el = time.perf_counter() - t0
            best = el if best is None else min(best, el)
        return best, res

    with torch.no_grad():
        t_vae, first_frame = timeit(lambda: model._encode_input_image_latents_tensor(input_image=img, tiled=False))
        latent_t = (spec.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
        latent_h = int(img.shape[-2]) // model.vae.upsampling_factor
        latent_w = int(img.shape[-1]) // model.vae.upsampling_factor
        latents_video = torch.randn((1, model.vae.model.z_dim, latent_t, latent_h, latent_w), dtype=torch.float32).to(device=dev, dtype=dt)
        latents_video[:, :, 0:1] = first_frame.clone()
        latents_action = torch.randn((1, spec.action_horizon, model.action_expert.action_dim), dtype=torch.float32).to(device=dev, dtype=dt)
        proprio = inputs["proprio"].to(device=dev, dtype=dt)
        context = inputs["context"].to(device=dev, dtype=dt)
        context_mask = inputs["context_mask"].to(device=dev, dtype=torch.bool)
        t_ctx, (context, context_mask) = timeit(lambda: model._append_proprio_to_context(context, context_mask, proprio), 1)
        fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
        schedule = model.infer_video_scheduler.build_inference_schedule(num_inference_steps=steps, device=dev, dtype=dt)
        timestep_video = schedule[0][0].unsqueeze(0).to(dtype=dt, device=dev)
        t_predit, video_pre = timeit(
            lambda: model.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        t_masks, attention_masks = timeit(
            lambda: model._build_mot_attention_masks(
                video_seq_len=video_seq_len,
                action_seq_len=latents_action.shape[1],
                video_tokens_per_frame=tokens_per_frame,
                device=dev,
            ),
            1,
        )
        t_vmasks, video_attention_mask = timeit(
            lambda: model._build_video_attention_mask(video_seq_len=video_seq_len, video_tokens_per_frame=tokens_per_frame, device=dev),
            1,
        )
        t_prefill, kv_cache = timeit(
            lambda: model.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=video_attention_mask,
            )
        )
        act_sched = model.infer_action_scheduler.build_inference_schedule(num_inference_steps=steps, device=dev, dtype=dt)
        timestep_action = act_sched[0][0].unsqueeze(0).to(dtype=dt, device=dev)

        def one_call():
            return model._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=kv_cache,
                attention_mask=attention_masks,
                video_seq_len=video_seq_len,
            )

        t_action_step, _pred = timeit(one_call)
        # Same calls but 10 back-to-back, i.e. what the denoise loop actually does.
        t_loop10, _l = timeit(lambda: [one_call() for _ in range(10)])
        out.update(
            {
                "vae_encode_s": round(t_vae, 4),
                "append_proprio_s": round(t_ctx, 4),
                "video_pre_dit_s": round(t_predit, 4),
                "mot_masks_s": round(t_masks, 4),
                "video_mask_s": round(t_vmasks, 4),
                "video_prefill_s": round(t_prefill, 4),
                "one_action_step_s": round(t_action_step, 4),
                "action_loop10_per_step_s": round(t_loop10 / 10.0, 4),
                "video_seq_len": video_seq_len,
                "tokens_per_frame": tokens_per_frame,
                "kv_cache_type": type(kv_cache).__name__,
                "kv_cache_entries": (sum(1 for e in kv_cache if e is not None) if isinstance(kv_cache, (list, tuple)) else len(kv_cache)),
                "reps": reps,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(args.threads))
    torch.set_grad_enabled(False)
    dtype = torch_dtype(args.dtype)

    print(
        f"[smoke] model={args.model} weights={args.weights} dtype={args.dtype} " f"steps={args.steps} iters={args.iters} threads={torch.get_num_threads()}",
        flush=True,
    )
    print(f"[smoke] torch {torch.__version__} cuda_available={torch.cuda.is_available()} " f"RSS={rss_mib():.0f} MiB", flush=True)

    cfg = load_cfg(args.model, args.task)
    ns, spec = make_spec(args.model, cfg, args.task, action_horizon=args.action_horizon, num_inference_steps=args.steps, synthetic_seed=args.synthetic_seed)
    print(f"[smoke] spec {json.dumps(spec.__dict__, default=str)}", flush=True)

    facts = {}
    build_start = time.perf_counter()
    if args.weights == "random":
        model, build_s = build_random(args, cfg, dtype)
        facts["build_model_seconds"] = build_s
    else:
        model, facts = build_ckpt(args, cfg, dtype)
    facts["build_total_seconds"] = time.perf_counter() - build_start
    facts["rss_after_build_mib"] = rss_mib()
    print(f"[smoke] build: {json.dumps({k: v for k, v in facts.items()}, default=str)}", flush=True)
    print(f"[smoke] params: {json.dumps(count_params(model))}", flush=True)
    print(f"[smoke] model.device={model.device} model.torch_dtype={model.torch_dtype} " f"RSS={rss_mib():.0f} MiB", flush=True)

    inputs = prepare_inputs(spec, torch.device("cpu"))
    inputs = {k: (v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inputs.items()}
    for name in ("video_expert", "action_expert", "vae"):
        sub = getattr(model, name, None)
        if sub is not None:
            seen = sorted({str(p.dtype) for p in sub.parameters()})
            print(f"[smoke] dtype {name}: {seen}", flush=True)

    kwargs: dict[str, Any] = dict(
        prompt=args.prompt,
        input_image=inputs["input_image"],
        action_horizon=spec.action_horizon,
        num_video_frames=spec.num_video_frames,
        proprio=inputs["proprio"],
        context=None if args.prompt is not None else inputs["context"],
        context_mask=None if args.prompt is not None else inputs["context_mask"],
        num_inference_steps=args.steps,
        seed=args.seed,
        rand_device="cpu",
        tiled=False,
    )

    if args.prompt is not None:
        print(f"[smoke] prompt={args.prompt!r}", flush=True)

    infer = model.infer_action_one_pass_future_cache

    def timed_run(steps: int, warmup: int, iters: int):
        kw = dict(kwargs)
        kw["num_inference_steps"] = steps
        times = []
        action_local = None
        for it in range(int(warmup) + int(iters)):
            start = time.perf_counter()
            with torch.no_grad():
                out = infer(**kw)
            elapsed = time.perf_counter() - start
            action_local = out["action"]
            if it >= int(warmup):
                times.append(elapsed)
        return times, action_local

    if args.steps_sweep:
        sweep = [int(x) for x in str(args.steps_sweep).split(",") if x.strip()]
        # One warm call at the largest size so allocator/page state is equal across points.
        timed_run(max(sweep), 0, 1)
        rows = []
        for steps in sweep:
            times, act_s = timed_run(steps, max(1, int(args.warmup)), max(2, int(args.iters)))
            row = {
                "steps": steps,
                "mean_s": round(float(np.mean(times)), 4),
                "min_s": round(float(np.min(times)), 4),
                "all_s": [round(t, 4) for t in times],
                "cksum": round(float(act_s.double().sum()), 6),
            }
            rows.append(row)
            print(f"[smoke] SWEEP {json.dumps(row)}", flush=True)
        if len(rows) >= 2:
            n0, n1 = rows[0]["steps"], rows[-1]["steps"]
            slope = (rows[-1]["mean_s"] - rows[0]["mean_s"]) / float(n1 - n0)
            print(f"[smoke] SLOPE s/step={slope:.4f} fixed={rows[0]['mean_s'] - slope * n0:.4f} " f"(from steps {n0}->{n1})", flush=True)
        print("[smoke] DONE", flush=True)
        return

    timings: list[float] = []
    action = None
    for it in range(int(args.warmup) + int(args.iters)):
        start = time.perf_counter()
        with torch.no_grad():
            out = infer(**kwargs)
        elapsed = time.perf_counter() - start
        action = out["action"]
        if it >= int(args.warmup):
            timings.append(elapsed)
            print(f"[smoke] iter {it - int(args.warmup)}: {elapsed:.3f} s", flush=True)

    act = action.detach().float()
    stats = {
        "shape": list(map(int, act.shape)),
        "min": float(act.min()),
        "max": float(act.max()),
        "mean": float(act.mean()),
        "checksum_sum": float(act.sum()),
        "checksum_sum64": float(np.float64(act.double().sum())),
    }
    summary = {
        "model": args.model,
        "weights": args.weights,
        "dtype": args.dtype,
        "steps": args.steps,
        "threads": int(args.threads),
        "per_iter_seconds": timings,
        "mean_seconds": float(np.mean(timings)),
        "min_seconds": float(np.min(timings)),
        "action": stats,
        "rss_peak_mib": rss_mib(),
        "params": count_params(model),
        **{k: v for k, v in facts.items() if k != "model_paths"},
    }
    print("[smoke] SUMMARY " + json.dumps(summary, default=str), flush=True)
    print("[smoke] action[0] = " + json.dumps(act[0].tolist()[:3], default=str) + " ...", flush=True)

    if args.stage_timing:
        try:
            facts["stages"] = stage_timing(model, spec, inputs, args.steps)
            print("[smoke] STAGES " + json.dumps(facts["stages"], default=str), flush=True)
        except Exception as exc:  # keep the latency numbers even if replay drifts
            facts["stages_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[smoke] STAGES failed: {facts['stages_error']}", flush=True)

    if args.dump:
        dump_path = Path(args.dump)
        if not dump_path.is_absolute():
            dump_path = SCRIPTS.parent / dump_path
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "inputs": {k: v.cpu() for k, v in inputs.items()},
                "action": act.cpu(),
                "spec": spec.__dict__,
                "kwargs": {k: v for k, v in kwargs.items() if not torch.is_tensor(v)},
                "summary": summary,
            },
            str(dump_path),
        )
        print(f"[smoke] dumped -> {dump_path}", flush=True)

    print("[smoke] DONE", flush=True)


if __name__ == "__main__":
    main()
