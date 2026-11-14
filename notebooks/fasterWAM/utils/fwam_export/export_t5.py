#!/usr/bin/env python
"""umT5-xxl text encoder -> OpenVINO IR: fp16 weights, fp32 activations, fp16 `context` output, CPU.

The three FasterWAM graphs start at `context` (128 text tokens, plus one proprio token the runtime
appends).  The shipped artifacts carry that context for the evaluation set, which is why the
conversion and the runtime never needed the text encoder.  This script closes the loop
`instruction -> ids -> umT5 -> context` on the machine that runs the notebook.

Recipe, measured 6 Nov (notebook pages `fasterwam-t5-export`, `fasterwam-t5-gpu-precision`):

  * trace in fp32 and compress **only the weights** to fp16 at save time.  `.half()` on the encoder
    itself is numerically unsafe (fp16 activations, context cos 0.72 on one prompt), while fp16
    weights with fp32 activations reach cos 0.99999993 against the torch bank.
  * CPU only.  int8 weights give 0.9908 and int4 0.8788 -- both NO-GO on context accuracy -- and
    fp16 on an Intel GPU NaNs (24 baked `finfo.min` constants turn into -inf, plus f16 overflow of
    the residual stream).
  * the graph is static ``[1, 128]``: one prompt per call.  A batch of two fails with
    "tensor size is not equal to model".
  * the wrapper reproduces ``FastWAM.encode_prompt`` exactly, including the "zero every position
    past ``mask.sum()``" pad; without that pad the context differs from the shipped bank.

Only the text encoder is built.  Building the whole FasterWAM model (the 5 B video expert in fp32)
next to the 4.7 B encoder is about 45 GB of weights and gets the process OOM-killed on a 64 GB box,
so this script loads the encoder alone, through the same hash-checked registry the repository's
loader uses -- the graph is identical, only the video/action/vae experts are never instantiated.

The export gates itself: once the IR is on disk it is compiled on the CPU, run for every evaluation
prompt, and compared with ``artifacts/eval_contexts.npz`` (cos >= ``--min-cos``, default 0.9999).
The result lands in ``<out-dir>/t5_manifest.json``; the exit code is non-zero when the gate fails.

Usage:
    python export_t5.py --out-dir <notebook>/model/ir/fp16/t5 \\
        --samples <notebook>/artifacts/eval_samples.json \\
        --ctx-ref <notebook>/artifacts/eval_contexts.npz

The checkpoint (``models_t5_umt5-xxl-enc-bf16.pth``, 11.4 GB) and the tokenizer are fetched by the
repository's own loader on first run; ``--free-checkpoint`` deletes the 11.4 GB file once the ONNX is
written, so a tight disk never holds the checkpoint and the IR at the same time.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent  # <notebook>/utils/fwam_export
NOTEBOOK = HERE.parents[1]  # <notebook>
REPO = Path(os.environ.get("REPO", NOTEBOOK / "model" / "FasterWAM")).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
# `export_openvino` and the repo resolve their paths at import time.
os.environ.setdefault("REPO", str(REPO))
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(REPO / "checkpoints"))

import export_openvino as EX  # noqa: E402  (to_ir, rss_mb)
import run_infer_smoke as RIS  # noqa: E402  (the config recipe)

SEQ = 128  # tokens per prompt, baked into the graph
ONNX_DIRNAME = "onnx"


def log(*a) -> None:
    print("[t5]", *a, flush=True)


class T5Enc(nn.Module):
    """``ids [1,128] int64`` + ``mask [1,128] int64`` -> ``context [1,128,4096]``.

    Zeroes every position past ``mask.sum()`` and nothing else, which is what ``encode_prompt``
    does before the proprio token is appended.  The mask argument is only used to build attention
    and that pad; the encoder returns the full 128 positions either way.
    """

    def __init__(self, t5: nn.Module) -> None:
        super().__init__()
        self.t5 = t5

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        keep = mask.to(torch.bool)
        emb = self.t5(ids, keep)  # [1, L, 4096]
        pos = torch.arange(emb.shape[1], dtype=torch.int64, device=emb.device)[None, :]
        live = (pos < keep.sum(dim=1, keepdim=True)).to(emb.dtype)  # [1, L] 1.0 / 0.0
        return emb * live.unsqueeze(-1)


def cos64(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Cosine and max abs difference of two float arrays, computed in float64."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    d = np.abs(a - b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)), float(d.max())


def build_text_encoder(cfg):
    """Load the umT5-xxl encoder and the tokenizer the way the repository's loader does.

    ``loader._resolve_configs`` -> ``ModelConfig.download_if_necessary`` (Hugging Face) ->
    ``_load_registered_model`` (hash-checked, so a wrong or truncated file is refused).  The
    FasterWAM config decides the repository ids, so this cannot drift from the model itself.
    """
    from fasterwam.models.wan22.helpers import loader as L
    from fasterwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    model_id = str(cfg.model.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
    tokenizer_id = str(cfg.model.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"))
    seq_len = int(cfg.model.get("tokenizer_max_len", SEQ))
    if seq_len != SEQ:
        raise SystemExit(f"[t5] tokenizer_max_len is {seq_len}, the export bakes {SEQ}")
    # build_ckpt pins this to False: the shipped .pth files match the registry hashes, so taking the
    # DiffSynth redirect would fetch a different (safetensors) file than the one this gate measured.
    redirect = False
    _dit_cfg, text_cfg, _vae_cfg, tok_cfg = L._resolve_configs(model_id, tokenizer_id, redirect_common_files=redirect)
    text_cfg.download_if_necessary()
    tok_cfg.download_if_necessary()
    log(f"weights  : {text_cfg.path}")
    log(f"tokenizer: {tok_cfg.path}")
    t5 = L._load_registered_model(str(text_cfg.path), "wan_video_text_encoder", torch_dtype=torch.float32, device="cpu")
    tokenizer = HuggingfaceTokenizer(name=str(tok_cfg.path), seq_len=seq_len, clean="whitespace")
    return t5.eval(), tokenizer, Path(str(text_cfg.path))


def export_onnx(enc: nn.Module, ids: np.ndarray, mask: np.ndarray, out_dir: Path) -> dict:
    """Trace the encoder to ONNX in fp32 (static [1,128]) and report size, time and RSS."""
    onnx_dir = out_dir / ONNX_DIRNAME
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    onnx_path = onnx_dir / "t5" / "t5.onnx"
    onnx_path.parent.mkdir(parents=True)
    sample = (torch.from_numpy(ids[:1]), torch.from_numpy(mask[:1]))
    t0 = time.perf_counter()
    torch.onnx.export(
        enc, sample, str(onnx_path), opset_version=17, dynamo=False, input_names=["ids", "mask"], output_names=["context"], do_constant_folding=True
    )
    export_s = time.perf_counter() - t0
    _cur, peak = EX.rss_mb()
    mb = sum(f.stat().st_size for f in onnx_dir.rglob("*") if f.is_file()) / 1e6
    return {"onnx": onnx_path, "onnx_mb": round(mb, 1), "export_s": round(export_s, 1), "rss_peak_export_mb": peak}


def ov_gate(xml: Path, ids: np.ndarray, mask: np.ndarray, ref_ctx: np.ndarray, threads: int = 24) -> dict:
    """Run the exported IR on the CPU for every prompt and compare with the shipped contexts."""
    import openvino as ov

    core = ov.Core()
    t0 = time.perf_counter()
    cm = core.compile_model(str(xml), "CPU", {"INFERENCE_NUM_THREADS": int(threads)})
    compile_s = time.perf_counter() - t0
    out_name = cm.output(0)
    cosines, maxabs, times = [], [], []
    for i in range(ids.shape[0]):
        t0 = time.perf_counter()
        out = np.asarray(cm([ids[i : i + 1], mask[i : i + 1]])[out_name])
        times.append(time.perf_counter() - t0)
        c, m = cos64(out, ref_ctx[i : i + 1])
        cosines.append(c)
        maxabs.append(m)
    for _ in range(2):  # warm, then time a few prompts
        cm([ids[0:1], mask[0:1]])
    warm = []
    for _ in range(min(5, ids.shape[0])):
        t0 = time.perf_counter()
        cm([ids[0:1], mask[0:1]])
        warm.append(time.perf_counter() - t0)
    return {
        "prompts": int(ids.shape[0]),
        "cos_min": min(cosines),
        "cos_mean": sum(cosines) / len(cosines),
        "max_abs_max": max(maxabs),
        "ms_per_prompt": round(sum(warm) / len(warm) * 1e3, 1),
        "compile_s": round(compile_s, 1),
        "first_pass_s": round(sum(times), 1),
        "cos_per_prompt": [round(c, 9) for c in cosines],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, help="IR directory, e.g. <notebook>/model/ir/fp16/t5")
    p.add_argument("--samples", default=str(NOTEBOOK / "artifacts/eval_samples.json"))
    p.add_argument("--ctx-ref", default=str(NOTEBOOK / "artifacts/eval_contexts.npz"))
    p.add_argument("--limit", type=int, default=0, help="gate on the first N prompts only (0 = all)")
    p.add_argument("--min-cos", type=float, default=0.9999, help="gate: minimum context cosine")
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--keep-onnx", action="store_true", help="keep the fp32 ONNX scratch (22.8 GB) next to the IR")
    p.add_argument("--free-checkpoint", action="store_true", help="delete the 11.4 GB umT5 checkpoint once the ONNX exists (it re-downloads)")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_onnx = args.keep_onnx or os.environ.get("FWAM_KEEP_ONNX", "").strip().lower() in ("1", "true", "yes")

    prompts = json.loads(Path(args.samples).read_text())["prompts"]
    ref_ctx = np.load(args.ctx_ref)["context"].astype(np.float32)
    if ref_ctx.shape[0] != len(prompts):
        raise SystemExit(f"[t5] {len(prompts)} prompts but {ref_ctx.shape[0]} shipped contexts")
    if args.limit:
        prompts, ref_ctx = prompts[: args.limit], ref_ctx[: args.limit]
    log(f"{len(prompts)} prompts; reference contexts {ref_ctx.shape} from {args.ctx_ref}")

    # ---- the encoder alone (the FasterWAM config decides which files, the registry checks them)
    cfg = RIS.load_cfg("fasterwam", None)
    t0 = time.perf_counter()
    text_encoder, tokenizer, ckpt = build_text_encoder(cfg)
    log(
        f"built {type(text_encoder).__name__} in {time.perf_counter() - t0:.1f} s: "
        f"dim={text_encoder.dim} layers={text_encoder.num_layers} "
        f"params={sum(q.numel() for q in text_encoder.parameters()) / 1e9:.3f} B"
    )

    ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = np.asarray(ids, dtype=np.int64)
    mask = np.asarray(mask, dtype=np.int64)
    if ids.shape[1] != SEQ:
        raise SystemExit(f"[t5] tokenizer produced {ids.shape[1]} tokens, the graph is [{SEQ}]")
    log(f"ids {ids.shape} mask_sum {mask.sum(1)[:4].tolist()} ... " f"(min {int(mask.sum(1).min())}, max {int(mask.sum(1).max())})")

    enc = T5Enc(text_encoder).eval()
    for q in enc.parameters():
        q.requires_grad_(False)
    torch.set_grad_enabled(False)

    # ---- the wrapper must reproduce `encode_prompt`, whose output is what the artifacts ship
    with torch.no_grad():
        mine = np.asarray(enc(torch.from_numpy(ids[:1]), torch.from_numpy(mask[:1])).numpy())
    c, m = cos64(mine, ref_ctx[0:1])
    log(f"wrapper vs the shipped context (prompt 0): cos={c:.9f} max_abs={m:.3e}")
    if c < 0.9999999:
        raise SystemExit("[t5] the wrapper does not reproduce encode_prompt -- not exporting")
    del mine
    gc.collect()

    # ---- fp32 ONNX -> fp16-weight IR
    onnx = export_onnx(enc, ids, mask, out_dir)
    log(f"ONNX {onnx['onnx_mb'] / 1000:.2f} GB in {onnx['export_s']:.0f} s " f"(rss peak {onnx['rss_peak_export_mb'] / 1000:.1f} GB) -> {onnx['onnx']}")

    if args.free_checkpoint and ckpt.is_file():
        freed = ckpt.stat().st_size / 1e9
        ckpt.unlink()
        log(f"deleted the text checkpoint ({freed:.2f} GB) -- the loader re-downloads it on demand")
    elif ckpt.is_file():
        log(f"text checkpoint stays at {ckpt} ({ckpt.stat().st_size / 1e9:.2f} GB)")

    del enc, text_encoder
    gc.collect()
    info = EX.to_ir(onnx["onnx"], out_dir, "t5", fp16=True, compress=True, fp16_io=True)  # the context output is f16
    log(
        f"IR fp16: {info['xml'].name} ({info['bin_mb'] / 1000:.2f} GB, ops {info['n_ops']}, "
        f"save {info['save_s']:.0f} s, rss peak {max(info['rss_reload_mb'][1], onnx['rss_peak_export_mb']) / 1000:.1f} GB)"
    )
    if not keep_onnx:
        shutil.rmtree(out_dir / ONNX_DIRNAME, ignore_errors=True)
        log("removed the ONNX scratch (the IR is self-contained)")

    # ---- gate against the shipped bank
    log(f"gating the IR on the CPU over {len(prompts)} prompts (this is the slow part)")
    gate = ov_gate(info["xml"], ids, mask, ref_ctx, threads=args.threads)
    gate["min_cos"] = float(args.min_cos)
    gate["passed"] = bool(gate["cos_min"] >= float(args.min_cos))
    for i, value in enumerate(gate["cos_per_prompt"][:8]):
        log(f"  prompt[{i:2d}] cos={value:.9f}")
    log(
        f"gate: cos_min={gate['cos_min']:.9f} mean={gate['cos_mean']:.9f} "
        f"max_abs={gate['max_abs_max']:.3e} | {gate['ms_per_prompt']:.0f} ms/prompt "
        f"(compile {gate['compile_s']:.1f} s) | {'PASS' if gate['passed'] else 'FAIL'}"
    )

    manifest = {
        "unit": "t5",
        "xml": str(info["xml"]),
        "bin_gb": round(info["bin_mb"] / 1000, 3),
        "n_ops": int(info["n_ops"]),
        "precision": "fp16 weights, fp32 activations, fp16 context output",
        "runtime_device": "CPU",
        "seq_len": SEQ,
        "checkpoint": str(ckpt),
        "tokenizer": str(Path(str(tokenizer.name))) if hasattr(tokenizer, "name") else "",
        "onnx_mb": onnx["onnx_mb"],
        "export_s": onnx["export_s"],
        "to_ir_s": round(info["read_s"] + info["save_s"], 1),
        "rss_peak_gb": round(max(info["rss_reload_mb"][1], onnx["rss_peak_export_mb"]) / 1000, 2),
        "gate": gate,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "t5_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"wrote {out_dir / 't5_manifest.json'}")
    if not gate["passed"]:
        raise SystemExit(f"[t5] GATE FAILED: cos_min {gate['cos_min']:.9f} < {gate['min_cos']}")


if __name__ == "__main__":
    main()
