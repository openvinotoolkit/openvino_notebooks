#!/usr/bin/env python
"""Export FasterWAM inference units to OpenVINO IR, verify numerics, benchmark.

Repo `/home/clee30/disk2/models/fasterWAM/FasterWAM` is READ-ONLY: all adaptation lives
here, as `torch.nn.Module` wrappers + runtime monkeypatching (never repo edits).

Units
  action : ONE sparse-action-expert denoising step (0.56 B params, runs 10x per replan chunk)
           in : latents_action[1,32,7], t[1], context[1,129,4096] (proprio token pre-appended),
                8 fused video K/V pairs [1,294,3072]  (16 tensors)  -> 19 inputs
           out: pred[1,32,7]
           masks + cross-attn context mask are baked INSIDE the wrapper as additive fp32 consts.
  vae    : single-frame VAE encode  [1,3,1,224,448] -> [1,48,1,14,28]
           (bypasses WanVideoVAE38's python list/feat_cache loop by calling encoder+conv1+mu-scale)
           plus `--vae-fold` (DEFAULT ON since the 31 Oct promotion, `--no-vae-fold` = pre-promotion
           graph): 22 of its 30 convolutions are `Conv3D kernel=[3,ky,kx]` on a D=1 tensor behind a
           depth-only Pad, so the kd=0,1 taps multiply exact zeros -> slice the filter to the live
           depth tap and rebuild as kernel=[1,ky,kx].  233.0 -> 81.4 GMAC, 2430 -> 1946 ops,
           296 -> 107.8 MB bin, -58 % VAE stage; the transform itself lives in `vae_opt/fold_vae.py`
           (`--unit vae --no-vae-fold` reproduces `ir_pre_vae_fold/`).
  video  : 5 B video prefill -> 8 fused K/V pairs.  NOT exported here (see README note / notebook).
  video_slice : same wrapper as `video` but only the FIRST N blocks (default 3) -> the fused K/V
           produced inside the slice + the block-N hidden state (extra output, for verification).
           Cheap probe of ONNX/IR size, RSS, compile time and device kernel coverage before
           attempting the full 30-layer 5 B graph.

Surgery performed by monkeypatching (module-level in fasterwam.models.wan22.wan_video_dit):
  * precompute_freqs_cis -> real (cos,sin) table packed as [end, dim//2, 2] float32 (angles in fp64)
  * rope_apply           -> real-valued rotation (kills torch.view_as_complex + float64)
  * flash_attention      -> bool attn_mask converted to additive float (-1e4) before SDPA
  * sinusoidal_embedding_1d -> fp32 or fp64 accumulation (flag --sin64)
Latency flags (all **default 1** since the 29 Oct promotion, which is what `ir/` and `ir_robotwin/` were
exported with; pass `0` to reproduce the pre-promotion graphs exactly; each is bit-identical in fp16):
  * --rope2 1 -> `rope_apply` without pair gathering: out = x*cos + flip(x)*(-sin,+sin) with widened
    constant tables; 3 GPU kernels per RoPE application instead of 9.  -1.5 ms/chunk.
  * --rope3 1 -> **default 1 since the 30 Oct promotion** (overrides --rope2, which stays default 1 as the
    control path `--rope3 0`): the SAME rotation written in the
    half-major (GPT-NeOX) `rotate_half` form on the last axis of a rank-4 [B,S,N,head_dim] tensor,
    which is the only traced form whose OpenVINO pattern matchers fire `RoPEFusionGPTNEOX` (exec:
    one tuned `ocl::rope::opt` kernel instead of `strided_slice_ref` + 2 `generic_eltwise_ref`).
    Math-preserving, not bit-identical: the pair-major -> half-major change of basis `P` is folded
    into the ROWS of `self_attn.q/k` (weight + bias) and of `norm_q/norm_k.weight` (`_perm_qk` /
    `apply_rope3_basis`), so q/k leave the GEMM already in the rotated basis, `P` cancels in q.k^T,
    V is untouched, and the 1024/3072-wide RMS reduction is permutation-invariant.  CAVEAT: the
    exported video K/V INTERFACE is then in the permuted basis too -> a `--rope3` video IR and a
    `--rope3` action IR are a matched pair and must be promoted together.  Shipping since 30 Oct:
    -1.5 ms video prefill + -0.24 ms action loop = ~134.5 ms/chunk LIBERO (40.1x), 145.2 RoboTwin;
    accuracy unchanged (LIBERO rmse 0.05533, RoboTwin 0.02341/0.1137, cos_min >= 0.99999 both).
  * --tmod2 1 -> the video expert's `t_mod` producer runs on the 2 distinct token timesteps (M=2)
    instead of all 294, then gathers back (`pre_dit_tmod2` + `_RowDedup`).  -0.5 ms/chunk.
  * --txtp N  -> EXPERIMENTAL (default 0 = untouched shipping graph): re-lay-out the text-context
    projection `MatMul([1,129,4096],[4096,3072])` (0.88 ms at 540 us/GMAC because M=129 is odd) --
    see `_TxtProj` for the modes.  Not a default; measured on the 2-layer probe.
  * --action-int8 / --no-action-int8 -> **default ON since the 1 Nov promotion** (`--unit action` only).
    After the ONNX->OV conversion the exporter (1) runs the B1 split (`action_b1/` + `action_ctx/`,
    see `split_action.py`), (2) compresses the 219 constant-B MatMuls of `action_fp16` and the 201 of
    `action_b1` with NNCF weight-only INT8 (`INT8_SYM`, ratio 1.0, group_size -1, data-free) and
    (3) `os.replace`s the result over the SAME filenames (no runtime/flag change).  Bins halve
    (1117.8 -> 560.9 MB, 1007.9 -> 505.1 MB) and the GPU plugin folds the dequant into its
    `FullyConnectedCompressed` primitive: action_loop -48.7 ms iGPU / -7.1 ms B580 per chunk.
    ORDER MATTERS: the split parses the **fp16** graph, so it must run BEFORE compression; the split
    also needs `<ntfy/>`-free fp16 bytes, i.e. never re-run it on a compressed IR.
    `--no-action-int8` = pre-promotion behaviour (fp16 action IR, no `action_b1` written).
  * --video-int8 / --no-video-int8 -> **default ON since the 1 Nov promotion** (`--unit video` ONLY:
    never `video_slice`, never `action`, never `vae`).  The 5 B video prefill is weight-byte-bound
    too -- 287 of its 399 MatMuls take a 2-D `Constant` B and those weights ARE the whole 9.382 GB
    bin -- so the same data-free NNCF pass halves the payload (9.382 -> 4.697 GB) and the GPU plugin
    folds `Constant u8 -> Convert -> Subtract(zp) -> Multiply(scale) -> MatMul` into its
    `FullyConnectedCompressed` primitive (iGPU: `FullyConnected` 287 x 125.62 -> 0, `FCC`
    287 x 66.89 + `DynamicQuantize` 175 x 4.29; layer count 2622 -> 2770).  video prefill
    -50.1 ms iGPU / -16.9 ms B580 per chunk, compile 16.6 -> 4.8 s, peak RSS 18.7 -> 9.7 GB.
    The scope is declared and asserted, not assumed: the 148 OTHER weight-bearing nodes (147
    int64 `Gather` index/frequency tables + 1 fp16 `Convolution` filter) are named in NNCF's
    `ignored_scope` and must come out UNCOMPRESSED (`_weight_nodes`; they are not fp16 weights to
    begin with, they are simply none of the quantiser's business).  ORDER CONTRACT -- the video
    unit has NO split step, so it is export -> compress -> `os.replace` under the SAME filenames
    (the fp16 original is parked by rename next to itself and DELETED once the INT8 bytes verify;
    no extra disk, no archive -- `FWAM_KEEP_FP16_ARCHIVE=1` keeps it instead).
    Compare the action unit, whose order is export -> B1 split -> compress: the split parses the
    **fp16** graph, so compression must never run first there.  `--no-video-int8` reproduces the
    fp16 graph (9381.8 MB) exactly.
  * --ctxpad N -> EXPERIMENTAL (default 0 = untouched shipping graph): pad the TEXT/CONTEXT STREAM
    once, inside `pre_dit`, from L=129 to the next multiple of 8 (136) and extend the additive
    cross-attention mask with -1e4 for the pad keys.  That converts ALL 57 odd-M GEMMs of the video
    unit (`1x129x3072 @ 3072x3072`, 4.74 ms = the text projection's 2nd Linear + the k and v
    projections of all 28 cross-attentions, all reading the same A operand) to M=136, and subsumes
    `--txtp` (whose mode 1 no-ops at M % 8 == 0).  Graph inputs/outputs are untouched.  See
    `_ctxpad_pair` / `_pre_dit_ctxpad`.  Not a default; measured on the 2-layer probe + full export.
  * --modg 1  -> needs --tmod2: keep `t_mod` as ONE ROW PER LATENT FRAME ([1,f,6,C]) and have
    `modulate` / `GateModule.forward` broadcast it against x reshaped to (B,f,tpf,C).  Deletes the 28
    per-block `modulation + t_mod` adds over [1,294,6,3072], the six per-token modulation slices per
    block and the per-token modulation reads; -7.8 ms/chunk (162.0 ms LIBERO chunk vs 171.6).
    Auditable: the exported IR then contains NO op of shape 1x294x6x3072.
  Offline gate for the last two: scripts/tmod2_test.py (no checkpoint needed).
and in fasterwam.models.wan22.wan_video_vae:
  * AvgDown3D.forward -> drops the degenerate `mean(dim=2)` when group_size == 1 (a no-op average
    that the intel_gpu plugin of OpenVINO 2026.5 cannot select a reduce kernel for, which made the
    whole VAE IR fail to compile on the Arc B580).  Bit-identical to the repo path.

Usage (always via scripts/ovrun.sh so the venv+OpenVINO env is loaded):
  scripts/ovrun.sh python .../export_openvino.py --unit action --dry-run
  scripts/ovrun.sh python .../export_openvino.py --unit action --dtype fp32 --verify --bench
  scripts/ovrun.sh python .../export_openvino.py --unit vae   --dtype fp16 --verify --bench
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rss_mb() -> tuple:
    """(current RSS MB, peak RSS MB) of this process, from /proc/self/statm + getrusage."""
    cur = 0.0
    try:
        with open("/proc/self/statm") as fh:
            cur = int(fh.read().split()[1]) * 4096 / 1e6
    except Exception:  # noqa: BLE001
        pass
    import resource

    return round(cur, 1), round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


# Bundled with the openvino_notebooks fasterWAM tutorial: the export toolchain lives next to
# this file, the PyTorch source is the clone the notebook fetches into `model/FasterWAM`
# (REPO env wins, else cwd - the runner sets cwd to the clone).
SCRIPTS = Path(__file__).resolve().parent
REPO = Path(os.environ.get("REPO", Path.cwd())).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO / "experiments" / "latency"))

import measure_latency as ml  # noqa: E402  (repo harness, CPU-safe fns only)
import run_infer_smoke as RIS  # noqa: E402  (our model-build recipe)
import fasterwam.models.wan22.wan_video_dit as wvd  # noqa: E402
import fasterwam.models.wan22.wan_video_vae as wvv  # noqa: E402
from fasterwam.models.wan22.wan_video_vae import AvgDown3D, count_conv3d, patchify  # noqa: E402

NEG_INF_MASK = -1e4  # additive mask value for masked positions (finite => fp16-safe)
DEFAULT_CKPT = str(REPO / "checkpoints/hustvl/FasterWAM/libero/step_021700.pt")
DEFAULT_REF = str(SCRIPTS / "artifacts/smoke_real_ckpt_steps10.pt")
ORIG = {}

# A2 ("--tmod2"): scratch context shared between the `pre_dit` wrapper, the row-deduping module
# wrappers and the sinusoidal patch below.  Written by `pre_dit_tmod2` around the repo's own call.
TMOD: dict = {}


def _tmod_idx(n: int, tpf: int, device) -> torch.Tensor:
    """Constant gather index [0]*tpf + [1]*(n - tpf): frame-0 tokens -> row 0, all others -> row 1.

    Built from a python list so it lands in the graph as a plain ONNX Constant.  Do NOT build it
    with `repeat_interleave`: int-repeats crash torch.onnx.export's constant folder (see
    `rope_apply_v2`).
    """
    return torch.tensor([0] * tpf + [1] * (n - tpf), dtype=torch.long, device=device)


def _tmod_frame_idx(f: int, device) -> torch.Tensor:
    """Gather index giving ONE row per latent frame (A1, `--modg`): [0] + [1]*(f-1), length **f**.

    The deduped producer holds 2 rows (frame 0 -> t=0, any other frame -> t) and `token_timesteps` is
    frame-major, so frame g simply takes row `min(g, 1)`.  The length must be f, not n = f*tpf: a
    length-n index here leaves t_mod at [1,n,6,C] and silently switches the A1 consumers off, which is
    exactly the bug that made the first A1 export cost nothing (scripts/tmod2_test.py now asserts
    t_mod.shape == (1, f, 6, C), and `ov_profile.py find --shape 1,294,6,3072` must print nothing).
    Built from a python list so it lands in the graph as a plain ONNX Constant -- `repeat_interleave`
    crashes torch.onnx.export's constant folder (see `rope_apply_v2`).
    """
    return torch.tensor([0] + [1] * (f - 1), dtype=torch.long, device=device)


class _RowDedup:
    """Run a row-independent nn.Sequential on the 2 distinct timestep rows, gather back afterwards.

    Temporarily installed (per `pre_dit` call, in the instance __dict__) in place of
    `time_embedding` / `time_projection`.  Both are row-independent (each output row depends only on
    the same input row), so values are unchanged -- only the number of rows their GEMMs see is.
    `expand` says how far to gather the small result back:
      "tokens" (A2) -> n rows, so t_mod keeps the shape every consumer already expects;
      "frames" (A1) -> the n rows are collapsed to one row per latent frame (f rows) and the two
                      consumers (`modulate`, `GateModule.forward`, both patched under `--modg`)
                      broadcast against x reshaped to (B, f, tpf, C).  t_mod then costs f*6*C
                      instead of n*6*C bytes and never touches the 294-token axis again.
    Any shape that does not match `TMOD["expect"]` falls through to the untouched module, so an
    unexpected workload can never silently get different numbers.
    """

    def __init__(self, seq: nn.Module, expand: str = "tokens") -> None:
        self.seq = seq
        self.expand = expand

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n, tpf = TMOD["rows"]
        axis = {2: 0, 3: 1}.get(x.dim())  # (n, c) or (1, n, c)
        if not TMOD.get("on") or axis is None or x.shape[-1] != TMOD["expect"].get(x.dim()):
            return self.seq(x)
        if x.shape[axis] != n or tpf < 1 or n <= tpf:
            return self.seq(x)
        small = torch.cat([x.narrow(axis, 0, 1), x.narrow(axis, tpf, 1)], dim=axis)  # (…, 2, c)
        if self.expand == "frames":
            f = TMOD["f"]
            if f * tpf != n or f < 2:
                return self.seq(x)
            return self.seq(small).index_select(axis, _tmod_frame_idx(TMOD["f"], x.device))
        return self.seq(small).index_select(axis, _tmod_idx(n, tpf, x.device))


class _TxtProj:
    """`text_embedding` with the FIRST Linear re-laid out for skinny-M (`--txtp`, TASK 2a).

    Temporarily installed (per `pre_dit` call, in the instance __dict__, exactly like `_RowDedup`)
    in place of `WanVideoDiT.text_embedding`, which is
    `Sequential(Linear(4096, 3072), GELU(tanh), Linear(3072, 3072))`.  In the shipping video IR the
    first Linear is `MatMul(A[1,129,4096], Const[4096,3072]) -> FullyConnected` and costs 0.88 ms for
    1.62 GMAC = 540 us/GMAC, ~15x off the 60 us weight-streaming roofline and 13x worse per MAC than
    the sibling `1x294x3072 @ 3072x3072` (39.8 us/GMAC).  The reason is the ODD token count: with
    M = 129 clDNN's `jit:gemm:any` kernel has to reload the 25.2 MB weight table many times over
    (measured: padding M to 136 makes the SAME kernel do the SAME work in 87 us = 54 us/GMAC).  Only
    the *layout* of that one GEMM changes here -- the weights, the bias, the GELU and the second
    Linear are the repo's own objects, so the result is the same math on the same data (bit-identical
    in fp32 except for split-K, which re-associates the accumulator; both were checked on the real
    checkpoint, see `torch_slice_worst` in the export log).

    `mode` (0 = off, the shipping graph):
      1 pad M up to the next multiple of 8 with zeros (`cat`), slice the output back   [136 for 129]
      2 same padding, done with `F.pad` (ONNX `Pad` -> clDNN `border_gpu_ref` instead of `Concat`)
      3 split K in 2 -> two MatMuls of K=2048 summed
      4 split K in 4 -> four MatMuls of K=1024 summed
      5 transposed product `(W[N,K] @ x^T[K,M])^T`
      6 squeeze the batch dim so the operand is 2-D `129x4096` (different kernel-selection path)
      7 1x1 `Convolution` form (`x` viewed as NCHW `[1,K,M,1]`, weight `[N,K,1,1]`)
      8 pad M up to the next multiple of 32 (160 for 129), like mode 1
    Every mode is guarded on the exact shape/structure of the module and the input; anything else
    runs `self.seq(x)`, the untouched repo expression.
    """

    def __init__(self, seq: nn.Module, mode: int) -> None:
        self.seq = seq
        self.mode = int(mode)

    def _applicable(self, x: torch.Tensor) -> bool:
        seq = self.seq
        if self.mode == 0 or x.dim() != 3 or int(x.shape[0]) != 1 or len(seq) != 3:
            return False
        lin0, act, lin2 = seq[0], seq[1], seq[2]
        return (
            isinstance(lin0, nn.Linear)
            and lin0.bias is not None
            and type(act).__name__ == "GELU"
            and isinstance(lin2, nn.Linear)
            and int(lin0.out_features) == int(lin2.in_features)
            and int(x.shape[2]) == int(lin0.in_features)
            and int(x.shape[1]) >= 8
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.seq
        if not self._applicable(x):
            return seq(x)  # verbatim repo expression
        lin0, act, lin2 = seq[0], seq[1], seq[2]
        w, bias, m = lin0.weight, lin0.bias, int(x.shape[1])
        mode = self.mode
        if mode in (1, 2, 8):
            align = {1: 8, 2: 8, 8: 32}[mode]
            pad = (-m) % align
            if pad == 0:
                return seq(x)
            if mode == 2:
                xp = F.pad(x, (0, 0, 0, pad))
            else:  # constant zeros -> clDNN `Concat`
                z = torch.zeros((int(x.shape[0]), pad, int(x.shape[2])), dtype=x.dtype, device=x.device)
                xp = torch.cat([x, z], dim=1)
            return lin2(act(F.linear(xp, w, bias).narrow(1, 0, m)))
        if mode in (3, 4):
            parts, k = (2 if mode == 3 else 4), int(w.shape[1])
            if k % parts:
                return seq(x)
            kh = k // parts
            y = None
            for i in range(parts):
                part = F.linear(x.narrow(2, i * kh, kh), w.narrow(1, i * kh, kh))
                y = part if y is None else y + part
            return lin2(act(y + bias))
        if mode == 5:
            return lin2(act(torch.matmul(w, x.transpose(1, 2)).transpose(1, 2) + bias))
        if mode == 6:
            return lin2(act(F.linear(x.squeeze(0), w, bias).unsqueeze(0)))
        if mode == 7:
            xc = x.transpose(1, 2).unsqueeze(-1)  # [B,K,M,1]
            y = F.conv2d(xc, w.reshape(w.shape[0], w.shape[1], 1, 1), bias)
            return lin2(act(y.squeeze(-1).transpose(1, 2)))  # [B,M,N]
        return seq(x)


def _pre_dit_txtp(mode: int):
    """`pre_dit` wrapper that shadows `text_embedding` with `_TxtProj` for the duration of the call.

    Wraps whatever `WanVideoDiT.pre_dit` currently is, so it composes with `pre_dit_tmod2` (install
    this one *after* that one).  The shadow lives in the instance __dict__ only while the original
    `pre_dit` runs and is removed in `finally`, so nothing else in the process -- the fp32 reference
    pass, the half trace, later units -- ever sees the re-laid-out module.
    """
    prev = wvd.WanVideoDiT.pre_dit

    def pre_dit_txtp(self, *args, **kwargs):
        te = getattr(self, "text_embedding", None)
        if not isinstance(te, nn.Sequential):
            return prev(self, *args, **kwargs)
        object.__setattr__(self, "text_embedding", _TxtProj(te, mode))
        try:
            return prev(self, *args, **kwargs)
        finally:
            self.__dict__.pop("text_embedding", None)  # look up through to _modules again

    pre_dit_txtp.__name__ = f"pre_dit_txtp{mode}"
    return pre_dit_txtp


TXTP_MODES = {
    1: "pad M to next x8 via cat",
    2: "pad M to next x8 via F.pad",
    3: "split K into 2",
    4: "split K into 4",
    5: "transposed (W @ x^T)^T",
    6: "2-D operand (squeeze batch)",
    7: "1x1 Convolution",
    8: "pad M to next x32",
}

CTXPAD_ALIGN_DEFAULT = 8  # `--ctxpad 1` -> pad the text stream to the next multiple of this


def _ctxpad_pair(ctx, msk, align: int):
    """Pad a `[B,L,D]` context and its `[B,L]` mask to L' = ceil(L/align)*align (`--ctxpad`).

    Returns `(ctx, msk)` UNCHANGED (identity, so the caller can tell) unless the pad applies.  The
    rules that make it a numerical no-op:
      * `context` pad rows are zeros.  They still come out of `text_embedding` as a constant
        (bias -> GELU -> Linear), so the 56 per-block key/value GEMMs do real arithmetic on them,
        but no downstream op reduces across the token axis, so their value cannot leak.
      * the mask pad value means "attend to nothing": `-NEG_INF_MASK`'s additive value (-1e4, finite
        so fp16-safe) for a float mask, `False` for a bool one -- exactly the two conventions
        `VideoPrefill.forward` / `flash_attention` use for masked text tokens.  Softmax then gets
        score + (-1e4) -> exp() underflows to exactly 0, so the kept positions are unchanged and the
        pad keys contribute 0*v_pad (proof + measurement: /tmp/p2b_unit.py).
      * the mask is REQUIRED: with `context_mask=None` `pre_dit` would synthesise an ALL-KEEP mask of
        the padded length, i.e. the pad keys would really be attended -> a different answer.  No mask,
        no pad.
      * a float mask is assumed to be the ADDITIVE form (0 = keep, -1e4 = drop), which is what every
        exported caller passes; a bool mask is converted downstream by `flash_attention`.
    Guards: 3-D context with a python-int token count, mask shape == context.shape[:2], float or bool
    mask, L >= 8, align >= 2.  Anything else returns the inputs verbatim, so an unexpected workload
    cannot silently get different numbers -- the graph stays exactly the shipping one.

    What it cannot be made of: `exp(-1e4)` is exactly 0, but the softmax/P@V kernels of the target
    device block by key count, so 136 keys vs 129 re-associates the fp16/fp32 sums.  Measured cost of
    that: ~1 ulp per element (/tmp/p2b_unit.py: fp32 1.3e-07, fp16 1.2e-04 on outputs of range 0.33,
    i.e. <= 1 ulp), against ~3e-02 if the pad keys were actually attended.  It is not removable -- any
    padding changes the key count -- and it is what `--verify`'s min_cos then measures on the GPU.

    Constants are built with `torch.zeros` / `torch.full` on a python-list-free, static shape, so
    they fold to ONNX `Const` nodes; nothing here uses `repeat_interleave` (which crashes
    torch.onnx.export's constant folder, see `rope_apply_v2`).
    """
    align = int(align)
    if align < 2 or not isinstance(ctx, torch.Tensor) or ctx.dim() != 3 or not msk_is_padgable(msk, ctx):
        return ctx, msk
    b, m = int(ctx.shape[0]), int(ctx.shape[1])
    pad = (-m) % align
    if pad == 0 or m < 8:
        return ctx, msk
    ctx_p = torch.cat([ctx, torch.zeros((b, pad, int(ctx.shape[2])), dtype=ctx.dtype, device=ctx.device)], dim=1)
    if msk.dtype == torch.bool:
        pad_row = torch.zeros((b, pad), dtype=torch.bool, device=msk.device)
    else:
        pad_row = torch.full((b, pad), NEG_INF_MASK, dtype=msk.dtype, device=msk.device)
    return ctx_p, torch.cat([msk, pad_row], dim=1)


def msk_is_padgable(msk, ctx) -> bool:
    """True if `msk` is a bool/float `[B, L]` mask matching `ctx`'s (B, L); None is NOT padgable."""
    return (
        isinstance(msk, torch.Tensor) and msk.dim() == 2 and tuple(msk.shape) == tuple(ctx.shape[:2]) and (msk.dtype == torch.bool or msk.is_floating_point())
    )


def _sparse_action_dit_cls():
    """`SparseActionDiT` or None -- imported lazily so a video-only export never needs the module."""
    try:
        from fasterwam.models.wan22.sparse_action_dit import SparseActionDiT

        return SparseActionDiT
    except Exception:  # pragma: no cover
        return None


def _pre_dit_ctxpad(align: int = CTXPAD_ALIGN_DEFAULT, cls: type | None = None):
    """`pre_dit` wrapper that pads the TEXT STREAM, not the graph interface (`--ctxpad`, TASK 2b).

    `cls` selects the expert whose `pre_dit` gets wrapped -- `wvd.WanVideoDiT` (default) or
    `sad.SparseActionDiT` (TASK D).  Both have the identical signature
    `(tokens, timestep, context, context_mask=None)` so the same kw/positional binding (idx 2/3)
    applies, and both feed the padded pair through the SAME consumer
    (`context_mask.unsqueeze(1).expand(-1, seq, -1)` -> `Mot._apply_expert_post_block` ->
    `block.cross_attn(..., ctx_mask=mask.unsqueeze(1))` -> additive-mask `flash_attention`), which is
    what makes the pad a no-op for either expert.  `patch_all` installs it on BOTH classes: patching
    the expert that is not being exported is inert, so one flag covers both units.

    Why: in the shipping video IR the context token count is ODD (L = 129), and clDNN's
    `jit:gemm:any` re-streams the weight table for odd/skinny M.  That costs 0.88 ms on the one
    `1x129x4096 @ 4096x3072` text projection (`--txtp`, TASK 2a) and 4.74 ms across the 57
    `1x129x3072 @ 3072x3072` GEMMs (83 us each against a 18.87 MB / ~44 us weight-streaming bound):
    `text_embedding`'s second Linear plus the key AND value projections of the cross-attention of
    all 28 video blocks -- every one of them reads the SAME `pre["context"]` tensor as its A operand
    (`ov_profile trace -m ir/video/video_fp16.xml --name /MatMul_10 --up 4` shows all 57 tracing
    back to /Add_5 <- /MatMul_2 <- the `context` Parameter).  Padding the stream ONCE at its entry
    point therefore converts all 57 GEMMs to M=136 and pays the padding glue exactly once.

    The pad is created INSIDE the graph, on `pre_dit`'s own arguments, so the exported unit keeps its
    exact interface -- `context [1,129,4096]` / `context_mask [1,129]` in, the 16 fused K/V
    `[1,294,3072]` out -- and `ov_runtime.py` / `ov_eval_ov.py` / the ACTION unit (a different class,
    `SparseActionDiT`, with its own `pre_dit`) need no change or re-export.

    What is padded: `context` -> [1,136,4096] (so BOTH Linear of `text_embedding` see M=136, which
    makes `--txtp` redundant: `_TxtProj` mode 1 already falls through when M % 8 == 0) and
    `context_mask` -> [1,136] with -1e4 (or False).  `pre_dit` then does its own
    `context_mask.unsqueeze(1).expand(-1, f*h*w, -1)` on the padded row, so the additive mask that
    every block's cross-attention adds to its scores is [1,1,294,136] and excludes the pad keys.

    Why this is exact: the cross-attention path (`CrossAttention.forward`, mot.py:145-152 ->
    wan_video_dit.py:215-220) applies no RoPE and no reduction over keys -- RoPE lives only in
    `_build_expert_attention_io` on the 294 video tokens, `norm_k`/`norm_q` are RMSNorms over the
    channel axis, the SDPA scale is 1/sqrt(head_dim), and `_fuse_video_kv` (the 16 graph outputs)
    caches the SELF-attention k/v.  Nor does the padded stream ever meet the `--modg` modulation
    guards: `modulate`/`GateModule.forward` are only called on the 294-token video stream
    (`block.norm3(context)`'s consumer is a plain LayerNorm), and `_frame_split` additionally keys on
    `mod.shape[1] == f` and `x.shape[1] == f*tpf`, which 136 rows can neither match nor defeat.

    Wraps whatever `cls.pre_dit` currently is, so it composes with `pre_dit_tmod2` and
    `pre_dit_txtp` in either install order (the padded M reaches `_TxtProj` either way).  If the
    arguments do not look as expected, `prev()` is called with the arguments verbatim.
    """
    cls = wvd.WanVideoDiT if cls is None else cls
    prev = cls.pre_dit

    def pre_dit_ctxpad(self, *args, **kwargs):
        ctx_kw, msk_kw = "context" in kwargs, "context_mask" in kwargs
        ctx = kwargs["context"] if ctx_kw else (args[2] if len(args) > 2 else None)
        msk = kwargs["context_mask"] if msk_kw else (args[3] if len(args) > 3 else None)
        ctx_p, msk_p = _ctxpad_pair(ctx, msk, align)
        if ctx_p is ctx:  # nothing to pad -> verbatim call
            return prev(self, *args, **kwargs)
        a, kw = list(args), dict(kwargs)
        for name, pos, val, was_kw in (("context", 2, ctx_p, ctx_kw), ("context_mask", 3, msk_p, msk_kw)):
            if val is None:
                continue
            if was_kw or len(a) <= pos:
                kw[name] = val
            else:
                a[pos] = val  # keep the caller's binding style
        return prev(self, *a, **kw)

    pre_dit_ctxpad.__name__ = f"pre_dit_ctxpad{align}"
    return pre_dit_ctxpad


# ---------------------------------------------------- rope3: pair-major -> half-major basis change
ROPE3: dict = {}  # filled by apply_rope3_basis(); {"head_dim": int, "n_modules": int, ...}


def head_halfmajor_perm(hd: int) -> torch.Tensor:
    """Gather index for the pair-major -> half-major (GPT-NeOX) reordering of ONE head.

    `perm[j]` is the SOURCE pair-major coordinate of half-major slot j:
        y = x[..., perm]  =>  y[0:hd/2] = x[0], x[2], ... x[hd-2]   (first element of each pair)
                              y[hd/2:] = x[1], x[3], ... x[hd-1]   (second element of each pair)
    In `y` the two elements of every rotation pair sit hd/2 apart, which is exactly what
    `rotate_half` (and therefore OV's `RoPEFusionGPTNEOX`) expects:
        rotate_half(y)[j]      = -y[j+hd/2] = -x[2j+1]   -> needs  cos_j / sin_j
        rotate_half(y)[j+hd/2] =  y[j]      =  x[2j]
    i.e. half-major RoPE with the doubled table `cat([c, c])` IS the shipped interleaved rotation
    written in the basis P:  rope_pair = P^-1 . rope_half . P.  Scripts/rope_v3_test.py pins this
    against an fp64 reference (and against a deliberately wrong permutation) before anything ships.
    """
    if hd < 4 or hd % 2:
        raise ValueError(f"head_dim {hd} cannot be re-laid-out in half-major form")
    return torch.tensor([2 * j if j < hd // 2 else 2 * (j - hd // 2) + 1 for j in range(hd)], dtype=torch.long)


def rope3_perm(d: int, hd: int) -> torch.Tensor:
    """Flat `[..., d]` gather index for a head-major layout of d//hd heads, each reordered by P."""
    if d % hd:
        raise ValueError(f"last dim {d} is not a whole number of {hd}-wide heads")
    p = head_halfmajor_perm(hd)
    return torch.cat([i * hd + p for i in range(d // hd)])


def rope3_permute(t: torch.Tensor, hd: int, inverse: bool = False) -> torch.Tensor:
    """Apply P (or P^-1) to the last axis of `t`, head-major with head width `hd`."""
    idx = rope3_perm(int(t.shape[-1]), hd)
    if inverse:
        idx = torch.argsort(idx)
    return t.index_select(-1, idx.to(t.device))


def _rope3_group_check(q, num_heads) -> None:
    """Prove the rope3 basis change cancels; called from the patched `flash_attention` at trace time.

    P cancels in q.k^T only if the attention groups the last axis into chunks that are whole
    multiples of the RoPE head width.  The repo has two call styles (`block.num_heads`, and
    `MoT.num_heads` where an action layer reads cached video K), so record the group sizes actually
    seen while tracing the real graph and refuse to export an unsupported one.  Pure python: runs
    once per call at trace time, never enters the graph.
    """
    if not num_heads:
        return
    group = int(q.shape[-1]) // int(num_heads)
    ROPE3.setdefault("attn_group_sizes", set()).add(group)
    hd = ROPE3.get("head_dim")
    if hd and group % int(hd):
        raise RuntimeError(
            f"--rope3: an attention groups the last axis into {group}, which is not a "
            f"multiple of the RoPE head width {hd}: the basis change P would NOT cancel "
            f"in q.k^T.  Refusing to export."
        )


def apply_rope3_basis(model) -> dict:
    """Fold P into the q/k projections + their RMSNorm gammas of every self-attention (`--rope3`).

    For a `SelfAttention` m:  q = norm_q(m.q(x))  and  k = norm_k(m.k(x))  are the ONLY tensors that
    RoPE touches (wan_video_dit.py:189-193, mot.py:201-205), so replacing
        W_q -> P W_q,  b_q -> P b_q,  gamma_q -> P gamma_q   (same for k)
    makes q/k arrive in the half-major basis for free: (P W)x + Pb = P(Wx + b) exactly, the RMSNorm
    denominator is a mean over the WHOLE attn_hidden_dim (permutation invariant), and its gamma is
    permuted with the rows.  P then cancels in q.k^T (both operands carry the same P), V and `o` are
    untouched, and no kernel is added anywhere.  Requires the attention to group the last axis into
    heads of exactly `hd` (else P would not cancel) -- asserted from the modules' own attributes.
    """
    if not hasattr(wvd, "SelfAttention"):
        raise RuntimeError("wan_video_dit.SelfAttention not available")
    reps, seen = [], set()
    for name, m in model.named_modules():
        if not isinstance(m, wvd.SelfAttention):
            continue
        hd, nh = int(m.attn_head_dim), int(m.num_heads)
        d = int(m.attn_hidden_dim)
        if d != hd * nh:
            raise RuntimeError(f"{name}: attn_hidden_dim {d} != num_heads {nh} * head_dim {hd}")
        idx = rope3_perm(d, hd)
        if int(idx.numel()) != d or torch.equal(idx, torch.arange(d)):
            raise RuntimeError(f"{name}: degenerate rope3 permutation")
        with torch.no_grad():
            for lin, gamma in ((m.q, m.norm_q.weight), (m.k, m.norm_k.weight)):
                lin.weight.data = lin.weight.data.index_select(0, idx).clone()
                if lin.bias is not None:
                    lin.bias.data = lin.bias.data.index_select(0, idx).clone()
                gamma.data = gamma.data.index_select(0, idx.to(gamma.device)).clone()
        seen.add(name)
        reps.append((name, nh, hd))
    if not reps:
        raise RuntimeError("apply_rope3_basis found no SelfAttention to re-base")
    # every module that is somebody's `self_attn` must have been re-based, or that attention would
    # get a permuted q against an unpermuted k (silent garbage, not a crash)
    missed = [n for n, mm in model.named_modules() if n.endswith("self_attn") and n not in seen]
    hds = sorted({r[2] for r in reps})
    if len(hds) != 1:
        raise RuntimeError(f"rope3 needs ONE head width across the model, got {hds}")
    ROPE3.clear()
    ROPE3.update(
        {
            "head_dim": hds[0],
            "n_modules": len(reps),
            "heads": sorted({r[1] for r in reps}),
            "missed_self_attn": missed,
            "first": reps[0][0],
            "last": reps[-1][0],
        }
    )
    return ROPE3


# ----------------------------------------------------------------------------- monkeypatches
def patch_all(sin64: bool = True, rope2: bool = False, tmod2: bool = False, modg: bool = False, txtp: int = 0, ctxpad: int = 0, rope3: bool = False) -> None:
    """Real-valued RoPE + additive float masks + optional fp64 sinusoidal embedding.

    `rope2=True` swaps in the pair-swap-free RoPE (see `rope_apply_v2`): same math, 3 GPU kernels
    per application instead of 9.  `rope3=True` (takes precedence over `rope2`; must be paired with
    `apply_rope3_basis(model)`, see `rope_apply_v3`): the same rotation in the half-major basis OV
    fuses into one kernel.  `tmod2=True` computes the video expert's per-token `t_mod` from
    its 2 distinct timesteps instead of all 294 (see `pre_dit_tmod2`).  `modg=True` (= A1, needs
    `tmod2`) additionally keeps `t_mod` as one row per *latent frame* and broadcasts it against the
    tokens reshaped to (B, f, tpf, C), which deletes the 28 per-block `modulation + t_mod` adds and
    the per-token scale/shift/gate reads behind them.  All default off so existing validated
    exports are unchanged.
    """
    if modg and not tmod2:
        raise ValueError("modg (A1) needs tmod2 (A2): the 2-row timestep producer comes first")
    if not ORIG:
        ORIG["rope"] = wvd.rope_apply
        ORIG["pcf"] = wvd.precompute_freqs_cis
        ORIG["fa"] = wvd.flash_attention
        ORIG["sin"] = wvd.sinusoidal_embedding_1d
        ORIG["avgdown"] = AvgDown3D.forward
        ORIG["predit"] = getattr(wvd.WanVideoDiT, "pre_dit", None)
        ORIG["sad_predit"] = getattr(_sparse_action_dit_cls(), "pre_dit", None)
        ORIG["modulate"] = getattr(wvd, "modulate", None)
        ORIG["gate"] = getattr(wvd.GateModule, "forward", None)
    acc = torch.float64 if sin64 else torch.float32

    # --- AvgDown3D: kill the size-1 ReduceMean that the intel_gpu plugin cannot lower --------
    # Repo tail is `x = x.view(B, out_c, group_size, T, H, W); return x.mean(dim=2)`.  For the last
    # encoder downsample (640->640, factor 1) group_size == 1, so the average is a no-op, but OV's
    # GPU plugin dies on it: "Failed to select implementation for name:.../avg_shortcut/ReduceMean
    # type: reduce".  Reducing over a size-1 axis is the identity; group_size > 1 keeps the repo's
    # mean(dim=2) verbatim, so the VAE output is bit-identical to the repo path.
    def avgdown3d_forward(self: AvgDown3D, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        ft, fs, g = self.factor_t, self.factor_s, self.group_size
        Tt, Hh, Ww = T // ft, H // fs, W // fs
        x = x.view(B, C, Tt, ft, Hh, fs, Ww, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, Tt, Hh, Ww)
        x = x.view(B, self.out_channels, g, Tt, Hh, Ww)
        if g == 1:
            return x.view(B, self.out_channels, Tt, Hh, Ww)
        return x.mean(dim=2)

    def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
        # angles in fp64, then cos/sin in fp64, stored packed fp32: [end, dim//2, 2]
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
        ang = torch.outer(torch.arange(end, dtype=torch.float64), freqs)
        return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).to(torch.float32)

    def rope_apply(x, freqs, num_heads):
        b, s = x.shape[0], x.shape[1]
        hd = x.shape[-1] // num_heads
        xr = x.reshape(b, s, num_heads, hd // 2, 2)
        cs = freqs.reshape(s, 1, hd // 2, 2).to(x.dtype)
        c, sn = cs[..., 0], cs[..., 1]
        x0, x1 = xr[..., 0], xr[..., 1]
        out = torch.stack([x0 * c - x1 * sn, x0 * sn + x1 * c], dim=-1)
        return out.reshape(b, s, -1)

    def rope_apply_v2(x, freqs, num_heads):
        """Same rotation as `rope_apply`, written so the exporter emits 3 kernels, not 9.

        The pair-indexing of v1 (`xr[..., 0]`, `xr[..., 1]`, `cs[..., 0]`, `cs[..., 1]` and the
        `torch.stack`) traces to 4 Gathers + 4 muls + sub + add + 2 Unsqueeze + Concat.  Here the
        pair swap is one flip on the (size-2) pair axis and cos/sin are widened to full head width,
        so out = x*cos + swap(x)*(-sin,+sin) traces to Gather/Flip + Mul + Mul-Add:
            out[2i]   = x[2i]*cos_i   - x[2i+1]*sin_i
            out[2i+1] = x[2i+1]*cos_i + x[2i]*sin_i
        The two tables come from `freqs` (a constant), so they are built once per graph and shared
        by every block; only the flip + 2 eltwise ops are per (q, k) application.

        The widened tables are built with slice+cat, NOT `repeat_interleave`: an int-repeats
        `repeat_interleave` on the constant-folded table makes torch.onnx.export's
        `_jit_pass_onnx_constant_fold` die on the *upstream* traced-int
        `self.freqs[0][:f].view(f, 1, 1, -1)` of the RoPE table with
        "RuntimeError: shape '[3, 1, 1, 2]' is invalid for input of size 132" (repro: 2-block
        `--unit video_slice --rope2 1`).  slice+cat yields bit-identical values in fp32 and fp16
        (checked vs the shipping `rope_apply` on the real 3-D table, 294 tokens x 24 heads).
        """
        b, s = x.shape[0], x.shape[1]
        hd = x.shape[-1] // num_heads
        cs = freqs.reshape(s, 1, hd // 2, 2).to(x.dtype)  # [s,1,hd/2,2] = (cos,sin)
        c = torch.cat([cs[..., 0:1], cs[..., 0:1]], dim=-1)  # cos on both pair slots
        sgn_sin = torch.cat([-cs[..., 1:2], cs[..., 1:2]], dim=-1)  # (-sin,+sin)
        xp = x.reshape(b, s, num_heads, hd // 2, 2)  # packed pairs
        return (xp * c + torch.flip(xp, dims=(-1,)) * sgn_sin).reshape(b, s, -1)

    def rope_apply_v3(x, freqs, num_heads):
        """`rope_apply_v2`'s rotation in the HALF-MAJOR head basis -- one fused GPU kernel.

        `x` must already be in the basis P (see `apply_rope3_basis`), i.e. its head coordinates are
        ordered [all first elements of a rotation pair, all second elements].  Then the rotation is
        GPT-NeOX `rotate_half` on the LAST axis of a rank-4 tensor with [1,S,1,head_dim] tables:
            out[..., j]       = x[j]*cos_j     - x[j+hd/2]*sin_j
            out[..., j+hd/2]  = x[j+hd/2]*cos_j + x[j]*sin_j
        This is the one traced shape (rank-4, rotation on the last axis, table broadcast over the
        head axis) for which OpenVINO's `RoPEFusionGPTNEOX` fires: the micro in
        /tmp/fuse/micro2.py shows `rms` + `RoPE (ocl::rope::opt)` for it, versus `rms` +
        `strided_slice_ref` + 2 `generic_eltwise_ref` for the shipped v2 form.  Values are the
        shipped rotation expressed in a permuted basis (rope_v3_test.py), so the IR is equivalent --
        not bit-identical -- to `--rope2 1`.
        """
        b, s = x.shape[0], x.shape[1]
        hd = x.shape[-1] // num_heads
        half = hd // 2
        cs = freqs.reshape(s, 1, hd // 2, 2).to(x.dtype)  # (cos,sin) per pair
        cos, sin = cs[..., 0], cs[..., 1]  # [s,1,hd/2]
        # NOTE both details below, each of which the test in scripts/rope_v3_test.py has a negative
        # control for: (a) the tables are duplicated along the PAIR axis (`cat` of two [s,1,hd/2] ->
        # [s,1,hd]), NOT by stacking on the packed dim, which would interleave them into the pair-
        # major layout again; (b) `rotate_half` already carries the minus sign, so the sin table is
        # plain +sin duplicated (HF convention) -- reusing v2's `cat([-sin, +sin])` trick here double-
        # negates and rotates the wrong way.
        c = torch.cat([cos, cos], dim=-1).reshape(1, s, 1, hd)
        sn = torch.cat([sin, sin], dim=-1).reshape(1, s, 1, hd)
        x4 = x.reshape(b, s, num_heads, hd)  # half-major pairs
        rot = torch.cat([-x4.narrow(-1, half, half), x4.narrow(-1, 0, half)], dim=-1)
        return (x4 * c + rot * sn).reshape(b, s, -1)

    def flash_attention(q, k, v, num_heads, ctx_mask=None, compatibility_mode=True):
        if rope3:  # record/validate the q grouping (see _rope3_group_check)
            _rope3_group_check(q, num_heads)
        from einops import rearrange

        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        if ctx_mask is not None and ctx_mask.dtype == torch.bool:
            ctx_mask = (ctx_mask.to(q.dtype) - 1.0) * (-NEG_INF_MASK)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        return rearrange(x, "b n s d -> b s (n d)", n=num_heads)

    def sinusoidal_embedding_1d(dim, position):
        pos = position.type(acc)
        sinusoid = torch.outer(pos, torch.pow(torch.tensor(10000.0, dtype=acc), -torch.arange(dim // 2, dtype=acc, device=pos.device).div(dim // 2)))
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return x.to(position.dtype)

    rope_impl = rope_apply_v3 if rope3 else (rope_apply_v2 if rope2 else rope_apply)

    # --- A2: video-expert `t_mod` from 2 distinct timesteps instead of all 294 ----------------
    # `pre_dit` (wan_video_dit.py:536-550) builds token_timesteps = ones(B, f, tpf) * t and then
    # zeroes frame 0, so the M = B*f*tpf token timesteps hold exactly TWO distinct values: 0 for the
    # tpf frame-0 tokens, t for the other (f-1)*tpf.  sinusoidal_embedding_1d -> time_embedding ->
    # time_projection are all row-independent, so evaluating them on [0, t] and gathering back to
    # B*f*tpf rows is exact (same values, same fp64 accumulation) while t_mod keeps the shape and
    # dtype every consumer already expects.  In the shipped LIBERO video IR that replaces /MatMul
    # [294,3072]x[3072,18432] (16.7 GMAC, 1.39 ms profiled) + /Gemm_1 + /Gemm + /Mul_6 + the fp64
    # Einsum/Cos/Sin at M=294 with the same ops at M=2 plus three gathers, and turns t_mod from a
    # 10.84 MB per-call tensor into a 36 KB 2-row table + expansion (which is what unlocks the
    # per-block `modulation + t_mod` gather, A1).
    sin_impl = sinusoidal_embedding_1d

    def sinusoidal_embedding_1d_tmod2(dim, position):
        """`sinusoidal_embedding_1d` for the (2-valued) token-timestep call; else untouched."""
        n, tpf = TMOD.get("rows", (0, 0))
        if not TMOD.get("on") or int(dim) != TMOD["expect"].get(2) or position.dim() != 1 or int(position.shape[0]) != n or tpf < 1:
            return sin_impl(dim, position)
        small = torch.cat([position.narrow(0, 0, 1), position.narrow(0, tpf, 1)], dim=0)  # [0, t]
        return sin_impl(dim, small).index_select(0, _tmod_idx(n, tpf, position.device))

    orig_predit = ORIG["predit"]

    def pre_dit_tmod2(self, *args, **kwargs):
        """`WanVideoDiT.pre_dit` with the t_mod producer squeezed to M=2 (see the note above).

        Only the row structure of `token_timesteps` is assumed; the maths is the repo's own code, so
        any workload that is not (B == 1, f >= 2, fused-VAE per-token timesteps) takes the verbatim
        repo path.  `time_embedding` / `time_projection` are shadowed in the *instance* __dict__
        (nn.Module.__setattr__ refuses to reassign a registered submodule) so state_dict and the
        exported weights are untouched, and the shadow is removed again in `finally`.
        """
        x = kwargs.get("x", args[0] if args else None)
        ts = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        ok = (
            x is not None
            and ts is not None
            and bool(kwargs.get("fuse_vae_embedding_in_latents", False))
            and bool(getattr(self, "seperated_timestep", False))
            and int(x.shape[0]) == 1
            and int(x.shape[2]) >= 2
        )
        if ok:
            patch_h, patch_w = int(self.patch_size[1]), int(self.patch_size[2])
            tpf = (int(x.shape[3]) // patch_h) * (int(x.shape[4]) // patch_w)
            ok = tpf >= 1 and int(ts.numel()) == 1
        if not ok:
            return orig_predit(self, *args, **kwargs)
        TMOD["rows"] = (int(x.shape[2]) * tpf, tpf)
        TMOD["f"] = int(x.shape[2])
        TMOD["expect"] = {2: int(self.freq_dim), 3: int(self.hidden_dim)}
        held, shadowed = {}, {}
        for name in ("time_embedding", "time_projection"):
            held[name] = getattr(self, name)
            shadowed[name] = name in self.__dict__  # nn.Module keeps it in _modules instead
            expand = "frames" if (modg and name == "time_projection") else "tokens"
            object.__setattr__(self, name, _RowDedup(held[name], expand))
        TMOD["on"] = True
        try:
            return orig_predit(self, *args, **kwargs)
        finally:
            TMOD["on"] = False
            for name, mod in held.items():  # undo the shadow, exactly as found
                if shadowed[name]:
                    object.__setattr__(self, name, mod)
                else:
                    self.__dict__.pop(name, None)  # look up through to _modules again

    wvd.precompute_freqs_cis = precompute_freqs_cis
    wvd.rope_apply = rope_impl
    wvd.flash_attention = flash_attention
    wvd.sinusoidal_embedding_1d = sinusoidal_embedding_1d
    wvv.AvgDown3D.forward = avgdown3d_forward

    # --- A1 (`--modg`): keep `t_mod` per LATENT FRAME and broadcast it against reshaped tokens ----
    # MoT._split_modulation is rank-generic: with t_mod [1,f,6,C] it returns 6 chunks of [1,f,C], so
    # only its two consumers need to know that the token axis is now (f, tpf) instead of (f*tpf):
    # `modulate` (mot.py:197,203 via wan_video_dit) and `GateModule.forward` (mot.py:145,156, twice per
    # block).  Reshaping x to (B,f,tpf,C) turns the per-token scale/shift/gate into a [B,f,1,C]
    # broadcast, so the 28 per-block `(modulation + t_mod)` adds over [1,294,6,3072] (21.7 MB of traffic
    # each) and the six 1.8 MB per-token modulation reads per block all disappear; every element is
    # computed with the same fp16 multiply/add as before, so the result is bit-identical.
    if modg:

        def _frame_split(x: torch.Tensor, mod: torch.Tensor):
            """(B, f*tpf, C) + its (B, f, C) modulation -> ((B,f,tpf,C), (B,f,1,C)); None if not that.

            Keyed on shapes: the modulation has exactly TMOD["f"] rows (one per latent frame) while `x`
            still carries f*tpf tokens.  Per-token modulation (rows == token count, the `--modg 0` case)
            and the action expert's [1,1,C] broadcast return None and the caller uses the repo
            expression verbatim.  A shape-only guard cannot separate the video case (f=3, tpf=98, n=294)
            from an action-expert call that happens to have f*tpf == its own token count; that is safe
            because folding rows into (f, tpf) is exactly the per-token expansion there -- checked in
            scripts/tmod2_test.py, which pins it against an explicit `repeat_interleave` reference.
            """
            f = int(TMOD.get("f", 0))
            tpf = int(TMOD.get("rows", (0, 0))[1])
            if (
                x.dim() != 3
                or mod.dim() != 3
                or f < 2
                or tpf < 1
                or int(mod.shape[1]) != f
                or int(mod.shape[0]) != int(x.shape[0])
                or int(mod.shape[2]) != int(x.shape[2])
                or int(x.shape[1]) != f * tpf
            ):
                return None
            return (x.reshape(x.shape[0], f, tpf, x.shape[2]), mod.reshape(mod.shape[0], f, 1, mod.shape[2]))

        def modulate_modg(x, shift, scale):
            xs, sv = _frame_split(x, shift) or (None, None)
            xc, cv = _frame_split(x, scale) or (None, None)
            if xs is None or sv is None or xc is None or cv is None:
                return x * (1 + scale) + shift  # verbatim repo expression
            return (xs * (1 + cv) + sv).reshape(x.shape)

        def gate_modg(self, x, gate, residual):
            xs, gv = _frame_split(x, gate) or (None, None)
            rs, _ = _frame_split(residual, gate) or (None, None)
            if xs is None or gv is None or rs is None:
                return x + gate * residual  # verbatim repo expression
            return (xs + gv * rs).reshape(x.shape)

        wvd.modulate = modulate_modg
        wvd.GateModule.forward = gate_modg
    if tmod2:
        wvd.sinusoidal_embedding_1d = sinusoidal_embedding_1d_tmod2
        wvd.WanVideoDiT.pre_dit = pre_dit_tmod2
    if int(txtp):
        # must come after the tmod2 install: it wraps whatever pre_dit is now (see _pre_dit_txtp)
        print(f"[ov] txtp={txtp}: text_embedding first Linear -> {TXTP_MODES.get(int(txtp), '?')} " f"(default 0 = untouched graph)", flush=True)
        wvd.WanVideoDiT.pre_dit = _pre_dit_txtp(int(txtp))
    if int(ctxpad):
        # outermost: pads pre_dit's own `context`/`context_mask` arguments, so it reaches whichever
        # wrapper (txtp / tmod2 / the repo method) is installed below it.  Default 0 -> untouched.
        align = int(ctxpad) if int(ctxpad) >= 2 else CTXPAD_ALIGN_DEFAULT
        print(
            f"[ov] ctxpad={ctxpad}: context/context_mask padded to a multiple of {align} INSIDE the "
            f"graph, cross-attn mask extended with {NEG_INF_MASK:g} on BOTH experts "
            f"(WanVideoDiT + SparseActionDiT) (default 0 = untouched graph)",
            flush=True,
        )
        wvd.WanVideoDiT.pre_dit = _pre_dit_ctxpad(align)
        sad_cls = _sparse_action_dit_cls()
        if sad_cls is not None:  # TASK D: the ACTION expert has its own pre_dit, and
            sad_cls.pre_dit = _pre_dit_ctxpad(align, sad_cls)  # its text projection is the odd-M victim
    # `from .x import y` copies already made by other modules must be re-pointed too.
    import fasterwam.models.wan22.mot as mot_mod
    import fasterwam.models.wan22.sparse_mot as smot_mod
    import fasterwam.models.wan22.sparse_action_dit as sad_mod
    import fasterwam.models.wan22.wan_video_text_encoder as wte_mod

    for mod in (mot_mod, smot_mod):
        if hasattr(mod, "rope_apply"):
            mod.rope_apply = rope_impl
        if hasattr(mod, "flash_attention"):
            mod.flash_attention = flash_attention
    if hasattr(sad_mod, "sinusoidal_embedding_1d"):
        sad_mod.sinusoidal_embedding_1d = sinusoidal_embedding_1d
    if hasattr(wte_mod, "precompute_freqs_cis"):
        wte_mod.precompute_freqs_cis = precompute_freqs_cis
    if modg:  # `from .wan_video_dit import modulate` copies must be re-pointed too
        for mod in (mot_mod, sad_mod):
            if hasattr(mod, "modulate"):
                mod.modulate = modulate_modg
    if hasattr(wvd, "WanVideoDiT"):
        pass  # video expert uses module globals -> already patched


def rebuild_freqs(model) -> None:
    """Expert RoPE tables are plain attributes (not buffers), so they were built by the
    ORIGINAL complex function at construction time. Recompute them with the patched one."""
    ae = getattr(model, "action_expert", None)
    if ae is not None and hasattr(ae, "freqs"):
        ae.freqs = wvd.precompute_freqs_cis(int(ae.attn_head_dim), end=int(ae.freqs.shape[0]))
    ve = getattr(model, "video_expert", None)
    if ve is not None and hasattr(ve, "freqs"):
        end = int(ve.freqs[0].shape[0]) if isinstance(ve.freqs, tuple) else 1024
        ve.freqs = wvd.precompute_freqs_cis_3d(int(ve.attn_head_dim), end=end)


def unpatch_all() -> None:
    if ORIG:
        wvd.rope_apply = ORIG["rope"]
        wvd.precompute_freqs_cis = ORIG["pcf"]
        wvd.flash_attention = ORIG["fa"]
        wvd.sinusoidal_embedding_1d = ORIG["sin"]
        if ORIG.get("predit") is not None:
            wvd.WanVideoDiT.pre_dit = ORIG["predit"]
        if ORIG.get("sad_predit") is not None:
            _sad = _sparse_action_dit_cls()
            if _sad is not None:
                _sad.pre_dit = ORIG["sad_predit"]
        if ORIG.get("modulate") is not None:
            wvd.modulate = ORIG["modulate"]
        if ORIG.get("gate") is not None:
            wvd.GateModule.forward = ORIG["gate"]
        TMOD.clear()
        ROPE3.clear()
        if "avgdown" in ORIG:
            wvv.AvgDown3D.forward = ORIG["avgdown"]
        # rebuild module-level freqs tables so the unpatched path is pristine
        import fasterwam.models.wan22.mot as mot_mod
        import fasterwam.models.wan22.sparse_mot as smot_mod
        import fasterwam.models.wan22.sparse_action_dit as sad_mod

        mot_mod.rope_apply = ORIG["rope"]
        smot_mod.rope_apply = ORIG["rope"]
        sad_mod.sinusoidal_embedding_1d = ORIG["sin"]
        if ORIG.get("modulate") is not None:
            wvd.modulate = ORIG["modulate"]
            mot_mod.modulate = ORIG["modulate"]
            sad_mod.modulate = ORIG["modulate"]
        if ORIG.get("gate") is not None:
            wvd.GateModule.forward = ORIG["gate"]


# ------------------------------------------------------------------------------- model build
def build(args) -> tuple:
    """Build FasterWAM with the real checkpoint in fp32 (recipe from run_infer_smoke)."""
    cfg = RIS.load_cfg("fasterwam", getattr(args, "task", None))
    _ns, spec = RIS.make_spec("fasterwam", cfg, getattr(args, "task", None), action_horizon=None, num_inference_steps=args.steps, synthetic_seed=0)
    nargs = SimpleNamespace(ckpt=args.model_dir, load_text_encoder=False)
    model, facts = RIS.build_ckpt(nargs, cfg, torch.float32)
    model.eval()
    # torch.onnx.export re-enables grad while tracing, which makes `param.to(...)` inside the
    # repo's blocks a grad-tracking tensor the tracer refuses to bake as a constant.
    for _p in model.parameters():
        _p.requires_grad_(False)
    torch.set_grad_enabled(False)
    return model, spec, facts


# ------------------------------------------------------------------------------- wrappers
class ActionStep(nn.Module):
    """One sparse-action-expert denoising step; repo path reused, masks baked as constants."""

    def __init__(self, model, masks, ctx_mask, video_seq_len, cond_layers):
        super().__init__()
        self._m = [model]  # plain attr: keep 5 B video expert out of our state_dict
        self.video_seq_len = int(video_seq_len)
        self.cond_layers = list(cond_layers)
        # bool [S,S] -> additive fp32 [S,S] constants (export-friendly, no bool SDPA mask)
        add = [(m.to(torch.float32) - 1.0) * (-NEG_INF_MASK) for m in masks]
        for i, t in enumerate(add):
            self.register_buffer(f"mask_{i}", t, persistent=False)
        self.register_buffer("ctx_mask", ctx_mask.to(torch.bool), persistent=False)

    def forward(self, latents_action, t, context, *kvs):
        model = self._m[0]
        n = model.mot.num_layers
        cache: list = [None] * n
        for i, layer in enumerate(self.cond_layers):
            cache[layer] = {"k": kvs[2 * i], "v": kvs[2 * i + 1]}
        masks = [getattr(self, f"mask_{i}") for i in range(n)]
        return model._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=t,
            context=context,
            context_mask=self.ctx_mask,
            video_kv_cache=cache,
            attention_mask=masks,
            video_seq_len=self.video_seq_len,
        )


class VaeEncode1(nn.Module):
    """Single-frame Wan2.2-TI2V-5B VAE encode: [1,3,1,H,W] -> [1,48,1,H/16,W/16]."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self._n_conv = count_conv3d(vae.model.encoder)
        self.register_buffer("mean", vae.mean.clone().float(), persistent=False)
        self.register_buffer("inv_std", (1.0 / vae.std.clone().float()), persistent=False)

    def forward(self, x):
        xm = patchify(x, patch_size=2)  # [1,12,1,H/2,W/2]
        out, _, _ = self.vae.model.encoder(xm[:, :, :1, :, :], feat_cache=[None] * self._n_conv, feat_idx=[0])
        mu, _logvar = self.vae.model.conv1(out).chunk(2, dim=1)
        z = self.mean.shape[0]
        mu = (mu - self.mean.view(1, z, 1, 1, 1)) * self.inv_std.view(1, z, 1, 1, 1)
        return mu


class VideoPrefill(nn.Module):
    """Video-expert prefill: patchify + per-token time/text embedding + `layers` blocks.

    Reuses the repo's own MoT helpers (`_build_expert_attention_io`,
    `_mixed_attention_with_num_heads`, `_apply_post_with_optional_checkpoint`, `_fuse_video_kv`)
    in exactly the order `SparseMoT.prefill_video_cache` uses them, so `layers=None` (30) IS the
    full video unit and a smaller N is a strict prefix of the same graph.
    Outputs: fused K/V for every condition layer inside the slice, then the block-(N-1) hidden
    state (extra output used only for verification; the production graph drops it).
    """

    def __init__(self, model, video_mask: torch.Tensor, layers=None, out_hidden: bool = True):
        super().__init__()
        self._m = [model]  # keep the 5 B expert out of state_dict
        self.layers = None if layers is None else int(layers)
        self.out_hidden = bool(out_hidden)
        # bool [S,S] self-attn mask -> additive fp32 constant (export-friendly, no bool SDPA mask)
        self.register_buffer("vmask", (video_mask.to(torch.float32) - 1.0) * (-NEG_INF_MASK), persistent=False)

    def forward(self, latents, t, context, ctx_mask):
        model = self._m[0]
        ve, mot = model.video_expert, model.mot
        fuse = bool(getattr(ve, "fuse_vae_embedding_in_latents", False))
        # context mask arrives 1.0/0.0 (bool is awkward as an ONNX input) -> additive, in the
        # activation dtype (fp16 when the graph was traced with half weights; SDPA requires the
        # float attn_mask to match the query dtype exactly)
        ctx_add = (ctx_mask.to(latents.dtype) - 1.0) * (-NEG_INF_MASK)
        pre = ve.pre_dit(x=latents, timestep=t, context=context, context_mask=ctx_add, action=None, fuse_vae_embedding_in_latents=fuse)
        x = pre["tokens"]
        payload = {"context": pre["context"], "mask": pre["context_mask"]}
        n = int(mot.num_layers) if self.layers is None else min(int(self.layers), int(mot.num_layers))
        outs: list = []
        interval: list = []
        for layer_idx in range(n):
            block = ve.blocks[layer_idx]
            q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gc = mot._build_expert_attention_io(
                expert=ve, block=block, x=x, freqs=pre["freqs"], t_mod=pre["t_mod"]
            )
            interval.append({"k": k, "v": v})
            mixed = mot._mixed_attention_with_num_heads(q_cat=q, k_cat=k, v_cat=v, attention_mask=self.vmask.to(q.dtype), num_heads=mot.num_heads)
            x = mot._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gc,
                mixed_slice=mixed,
                context_payload=payload,
            )
            if layer_idx in mot.condition_layer_set:
                kv = mot._fuse_video_kv(layer_idx, interval)
                interval = []
                outs += [kv["k"], kv["v"]]
        return tuple(outs + ([x] if self.out_hidden else []))


def derive_action_ref(model, spec, inputs, seed: int = 1234) -> dict:
    """Replay infer_action_one_pass_future_cache up to the first action step (repo code, unpatched)."""
    dev = torch.device("cpu")
    dt = model.torch_dtype
    torch.manual_seed(seed)
    img = inputs["input_image"]
    first_frame = model._encode_input_image_latents_tensor(input_image=img, tiled=False)
    latent_t = (spec.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    lh = int(img.shape[-2]) // model.vae.upsampling_factor
    lw = int(img.shape[-1]) // model.vae.upsampling_factor
    latents_video = torch.randn((1, model.vae.model.z_dim, latent_t, lh, lw), dtype=torch.float32).to(dt)
    latents_video[:, :, 0:1] = first_frame.clone()
    latents_action = torch.randn((1, spec.action_horizon, model.action_expert.action_dim), dtype=torch.float32).to(dt)
    proprio = inputs["proprio"].to(dt)
    context = inputs["context"].to(dt)
    context_mask = inputs["context_mask"].to(torch.bool)
    context, context_mask = model._append_proprio_to_context(context, context_mask, proprio)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    sched = model.infer_video_scheduler.build_inference_schedule(num_inference_steps=10, device=dev, dtype=dt)
    t_video = sched[0][0].unsqueeze(0).to(dt)
    pre = model.video_expert.pre_dit(
        x=latents_video, timestep=t_video, context=context, context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse
    )
    vs = int(pre["tokens"].shape[1])
    tpf = int(pre["meta"]["tokens_per_frame"])
    masks = model._build_mot_attention_masks(video_seq_len=vs, action_seq_len=latents_action.shape[1], video_tokens_per_frame=tpf, device=dev)
    vmask = model._build_video_attention_mask(video_seq_len=vs, video_tokens_per_frame=tpf, device=dev)
    kv = model.mot.prefill_video_cache(
        video_tokens=pre["tokens"],
        video_freqs=pre["freqs"],
        video_t_mod=pre["t_mod"],
        video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_attention_mask=vmask,
    )
    asch = model.infer_action_scheduler.build_inference_schedule(num_inference_steps=10, device=dev, dtype=dt)
    t_action = asch[0][0].unsqueeze(0).to(dt)
    pred = model._predict_action_noise_with_cache(
        latents_action=latents_action,
        timestep_action=t_action,
        context=context,
        context_mask=context_mask,
        video_kv_cache=kv,
        attention_mask=masks,
        video_seq_len=vs,
    )
    cond = [i for i, e in enumerate(kv) if e is not None]
    return {
        "latents_action": latents_action.float().cpu(),
        "t": t_action.float().cpu(),
        "context": context.float().cpu(),
        "ctx_mask": context_mask.cpu(),
        "kv_k": [kv[i]["k"].float().cpu() for i in cond],
        "kv_v": [kv[i]["v"].float().cpu() for i in cond],
        "masks": [m.cpu() for m in masks],
        "video_seq_len": vs,
        "cond_layers": cond,
        "timesteps_all": asch[0].float().cpu(),
        "pred": pred.float().cpu(),
    }


def derive_vae_ref(model, inputs) -> dict:
    img = inputs["input_image"]  # [1,3,H,W] as fed to infer_action_*
    z = model._encode_input_image_latents_tensor(input_image=img, tiled=False)
    x = img.float().unsqueeze(2).cpu()  # [1,3,1,H,W] = the graph input
    return {"x": x, "z": z.float().cpu()}


# ------------------------------------------------------------------------------- metrics
def derive_video_ref(model, spec, inputs, keep_blocks: int = 8, seed: int = 1234) -> dict:
    """Unpatched repo video prefill + a spy on MoT._apply_expert_post_block, so each layer's
    output token state is captured for slice verification (the MoT path never calls block(...),
    so nn.Module hooks on the blocks would not fire)."""
    import types  # noqa: F401

    dev = torch.device("cpu")
    dt = model.torch_dtype
    captured: list = []
    cls = type(model.mot)
    orig_post = cls._apply_expert_post_block

    def spy(**kw):
        out = orig_post(**kw)
        captured.append(out.detach().float().cpu())
        return out

    model.mot._apply_expert_post_block = spy
    try:
        torch.manual_seed(seed)
        img = inputs["input_image"]
        first_frame = model._encode_input_image_latents_tensor(input_image=img, tiled=False)
        latent_t = (spec.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
        lh = int(img.shape[-2]) // model.vae.upsampling_factor
        lw = int(img.shape[-1]) // model.vae.upsampling_factor
        latents_video = torch.randn((1, model.vae.model.z_dim, latent_t, lh, lw), dtype=torch.float32).to(dt)
        latents_video[:, :, 0:1] = first_frame.clone()
        proprio = inputs["proprio"].to(dt)
        context = inputs["context"].to(dt)
        context_mask = inputs["context_mask"].to(torch.bool)
        context, context_mask = model._append_proprio_to_context(context, context_mask, proprio)
        fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
        sched = model.infer_video_scheduler.build_inference_schedule(num_inference_steps=10, device=dev, dtype=dt)
        t_video = sched[0][0].unsqueeze(0).to(dt)
        pre = model.video_expert.pre_dit(
            x=latents_video, timestep=t_video, context=context, context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse
        )
        vs = int(pre["tokens"].shape[1])
        tpf = int(pre["meta"]["tokens_per_frame"])
        vmask = model._build_video_attention_mask(video_seq_len=vs, video_tokens_per_frame=tpf, device=dev)
        kv = model.mot.prefill_video_cache(
            video_tokens=pre["tokens"],
            video_freqs=pre["freqs"],
            video_t_mod=pre["t_mod"],
            video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_attention_mask=vmask,
        )
    finally:
        try:
            del model.mot._apply_expert_post_block
        except Exception:  # noqa: BLE001
            pass
    cond = [i for i, e in enumerate(kv) if e is not None]
    return {
        "latents": latents_video.float().cpu(),
        "t": t_video.float().cpu(),
        "context": context.float().cpu(),
        "ctx_mask": context_mask.float().cpu(),  # fp32 1.0/0.0 graph input
        "ctx_mask_bool": context_mask.cpu(),
        "vmask": vmask.cpu(),
        "video_seq_len": vs,
        "cond_layers": cond,
        "kv_k": [kv[i]["k"].float().cpu() for i in cond],
        "kv_v": [kv[i]["v"].float().cpu() for i in cond],
        "blk_out": [t for t in captured[:keep_blocks]],
    }


def metrics(ref: torch.Tensor, got: torch.Tensor) -> dict:
    a = ref.detach().double().reshape(-1)
    b = got.detach().double().reshape(-1)
    d = (a - b).abs()
    denom = a.abs().clamp_min(1e-6)
    cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))
    amax = float(a.abs().max())
    return {
        "max_abs": float(d.max()),
        "max_rel": float((d / denom).max()),
        "rmse": float((d * d).mean().sqrt()),
        "cos": cos,
        "ref_absmax": amax,
        "rel_to_range": float(d.max() / max(amax, 1e-30)),
    }


def fmt(m: dict) -> str:
    return f"maxabs={m['max_abs']:.3e} maxrel={m['max_rel']:.3e} " f"maxrel_over_outrange={m['rel_to_range']:.3e} rmse={m['rmse']:.3e} cos={m['cos']:.9f}"


# ------------------------------------------------------------------------------- export/bench
def export(wrapper: nn.Module, sample: tuple, onnx_dir: Path, names: dict, tag: str) -> dict:
    # torch's ONNX exporter writes big initializers as ONE FILE PER TENSOR next to the .onnx,
    # so give each unit its own clean directory and count everything in it.
    onnx_dir = onnx_dir / tag
    for stale in onnx_dir.glob("*"):
        stale.unlink() if stale.is_file() else None
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{tag}.onnx"
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            wrapper, sample, str(onnx_path), opset_version=17, dynamo=False, input_names=names["in"], output_names=names["out"], do_constant_folding=True
        )
    export_s = time.perf_counter() - t0
    rss_cur, rss_peak = rss_mb()
    onnx_mb = sum(f.stat().st_size for f in onnx_dir.rglob("*") if f.is_file()) / 1e6
    n_files = sum(1 for f in onnx_dir.rglob("*") if f.is_file())
    return {"onnx": onnx_path, "onnx_mb": onnx_mb, "export_s": export_s, "n_files": n_files, "rss_after_cur_mb": rss_cur, "rss_peak_mb": rss_peak}


# ---- fp16 graph interfaces (--fp16-io, DEFAULT ON for fp16 IRs) -----------------------------
# `compress_to_fp16` only converts the WEIGHTS: the exported graph still declares f32 inputs and
# outputs, so the GPU plugin casts every bulk tensor at the boundary (16 video K/V outputs
# [1,294,3072] f32 -> 16 f32 action inputs that are re-cast to f16 on every one of the 10 steps)
# and the host casts/copies them between the units.  Declaring the bulk tensors f16 removes those
# conversions: -6.4 ms/chunk on an Arc B390 (250.2 -> 243.8), action chunk cos >= 0.9999999.
# Done with explicit Convert nodes, NOT PrePostProcessor (whose build() also constant-folds the
# graph and moved the bf16 T5 context cos_min 0.990 -> 0.964).  Applied right after `read_model`,
# so the VAE fold, the B1 split and the INT8 passes (which re-read the saved xml) all inherit it.
# Kept f32 on purpose: video `timestep` (fp16 spacing near t=1000 is 0.5) and the action unit's
# `latents_action` / `t` / `pred` (the host accumulates the Euler state in f32).
FP16_KEEP_IN = {"video": {"timestep"}, "action": {"latents_action", "t"}}
FP16_KEEP_OUT = {"action": {"pred"}}


def _iface_unit(tag: str) -> str:
    return "video" if tag.startswith("video") else tag


def set_fp16_interface(model, unit: str) -> dict:
    """Insert Convert nodes so every f32 input/output of `model` is f16, except the FP16_KEEP_* tensors.

    Names, order and shapes of the interface are unchanged (+1 op per converted tensor).  Returns
    {"in": [...], "out": [...]} with the names that became f16.
    """
    from openvino import Type
    from openvino import opset13 as ops

    keep_in, keep_out = FP16_KEEP_IN.get(unit, set()), FP16_KEEP_OUT.get(unit, set())
    changed = {"in": [], "out": []}
    for idx, prm in enumerate(list(model.get_parameters())):
        name = prm.get_output_tensor(0).get_any_name()
        if prm.get_output_element_type(0) != Type.f32 or name in keep_in:
            continue
        new = ops.parameter(prm.get_output_partial_shape(0), Type.f16, name=prm.get_friendly_name())
        new.output(0).get_tensor().set_names(set(prm.output(0).get_names()))
        conv = ops.convert(new, Type.f32)
        for tin in list(prm.output(0).get_target_inputs()):
            tin.replace_source_output(conv.output(0))
        model.replace_parameter(idx, new)
        changed["in"].append(name)
    for res in model.get_results():
        src = res.input_value(0)
        names = set(res.output(0).get_names()) or set(src.get_names())
        name = sorted(names)[0] if names else res.get_friendly_name()
        if src.get_element_type() != Type.f32 or name in keep_out:
            continue
        conv = ops.convert(src, Type.f16)
        src.get_tensor().set_names(set())
        conv.output(0).get_tensor().set_names(names)
        res.input(0).replace_source_output(conv.output(0))
        changed["out"].append(name)
    model.validate_nodes_and_infer_types()
    return changed


def feed_arrays(model, sample) -> list:
    """The sample tensors as numpy arrays, each in the element type the model DECLARES for that input."""
    out = []
    for prm, t in zip(model.inputs, sample):
        et = str(prm.get_element_type())
        dt = {"<Type: 'float16'>": np.float16, "<Type: 'float32'>": np.float32}.get(et)
        a = t.contiguous().numpy()
        out.append(a.astype(dt) if dt is not None and a.dtype != dt else a)
    return out


def to_ir(onnx_path: Path, ir_dir: Path, tag: str, fp16: bool, compress: bool = True, fp16_io: bool = False) -> dict:
    import openvino as ov

    core = ov.Core()
    rss_before = rss_mb()
    t0 = time.perf_counter()
    om = core.read_model(str(onnx_path))
    read_s = time.perf_counter() - t0
    rss_read = rss_mb()
    n_ops = len(om.get_ops())
    iface = set_fp16_interface(om, _iface_unit(tag)) if (fp16_io and fp16) else None
    ir_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_{'fp16' if fp16 else 'fp32'}"
    xml = ir_dir / f"{stem}.xml"
    t0 = time.perf_counter()
    ov.save_model(om, str(xml), compress_to_fp16=bool(compress))
    save_s = time.perf_counter() - t0
    bin_mb = (ir_dir / f"{stem}.bin").stat().st_size / 1e6 if (ir_dir / f"{stem}.bin").exists() else 0.0
    t0 = time.perf_counter()
    om = core.read_model(str(xml))
    read_ir_s = time.perf_counter() - t0
    rss_reload = rss_mb()
    return {
        "core": core,
        "model": om,
        "xml": xml,
        "n_ops": n_ops,
        "fp16_io": iface,
        "read_s": round(read_s, 2),
        "save_s": round(save_s, 2),
        "read_ir_s": round(read_ir_s, 2),
        "bin_mb": bin_mb,
        "stem": stem,
        "rss_read_model_mb": [rss_read[0], max(rss_read[1], rss_before[1])],
        "rss_reload_mb": [rss_reload[0], max(rss_reload[1], rss_read[1])],
    }


# ---- VAE depth-tap fold (DEFAULT ON for --unit vae since the 31 Oct promotion) ------------
# 22 of the VAE encoder's 30 convolutions are `Conv3D kernel=[3,ky,kx]` on a D=1 tensor that
# sits at depth index 2 of a `Pad([0,0,2,1,1],[0,0,0,1,1])`: the kd=0,1 taps multiply exact
# zeros (233.0 -> 81.4 GMAC, 296 -> 107.8 MB bin, -58 % on the VAE stage).  The transform is
# IR-level and lives in `vae_opt/fold_vae.py` so the standalone artifact generator and this
# exporter cannot drift apart; here we only wire it in after the ONNX->OV conversion.
def _vae_fold_lib():
    """Import `fold_vae.py` (the IR fold) from wherever it lives on this box."""
    import importlib.util

    here = Path(__file__).resolve().parent
    cands = [
        os.environ.get("VAE_FOLD_LIB"),
        str(here / "fold_vae.py"),
        str(here / "vae_opt" / "fold_vae.py"),
        str(here.parent / "vae_opt" / "fold_vae.py"),
        str(SCRIPTS / "fold_vae.py"),
        str(SCRIPTS.parent / "vae_opt" / "fold_vae.py"),
    ]
    for cand in cands:
        if cand and Path(cand).is_file():
            spec = importlib.util.spec_from_file_location("fold_vae", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, Path(cand)
    raise SystemExit("[ov] --vae-fold: cannot find fold_vae.py (looked in " f"{[c for c in cands if c]}); pass --no-vae-fold to export the unfolded graph")


def apply_vae_fold(info: dict, compress: bool = True) -> tuple:
    """Fold the VAE encoder's dead depth taps into the freshly converted IR, in place.

    Rewrites `<out-dir>/vae/vae_fp16.{xml,bin}` (same filenames -> no runtime/flag change)
    and returns (updated `to_ir` info dict, stats dict).  Asserts afterwards that no
    `Conv3D kernel[0]==3` on a depth-only-padded D=1 input survives.
    """
    import openvino as ov

    lib, lib_path = _vae_fold_lib()
    core, model, xml = info["core"], info["model"], Path(info["xml"])
    bin_path = xml.with_suffix(".bin")
    n_ops_src = len(model.get_ops())
    stats = lib.fold_graph(model, log=print)
    # NB: `xml`/`bin` were just written by to_ir() and `model` may be backed by a mmap of
    # that very .bin -- writing in place would SIGBUS the process.  Save next to it and
    # atomically rename, so the shipping filename also never exists in a half-written state.
    tmp_xml = xml.with_name(xml.stem + ".foldtmp.xml")
    tmp_bin = tmp_xml.with_suffix(".bin")
    t0 = time.perf_counter()
    ov.save_model(model, str(tmp_xml), compress_to_fp16=bool(compress))
    save_s = time.perf_counter() - t0
    os.replace(str(tmp_xml), str(xml))
    os.replace(str(tmp_bin), str(bin_path))
    del model
    om = core.read_model(str(xml))
    chk = lib.verify_model(om)  # asserts: no depth taps left
    out = dict(info, model=om, n_ops=n_ops_src, bin_mb=bin_path.stat().st_size / 1e6, vae_fold_save_s=round(save_s, 2))
    meta = {
        "lib": str(lib_path),
        "xml": str(xml),
        "macs_before_gmac": round(stats["mac_before"] / 1e9, 3),
        "macs_after_gmac": round(stats["mac_after"] / 1e9, 3),
        "mac_reduction": round(stats["mac_before"] / max(stats["mac_after"], 1e-9), 3),
        "macs_rereload_gmac": round(chk["macs_gmac"], 3),
        "convs_folded": stats["folded"],
        "convs_skipped": len(stats["skipped"]),
        "kd3_convs_remaining": chk["kd3_convs"],
        "depth_taps_left": chk["depth_taps_left"],
        "ops_before": n_ops_src,
        "ops_after": chk["n_ops"],
        "bin_mb": round(out["bin_mb"], 1),
        "save_s": round(save_s, 2),
    }
    # the reloaded graph must carry no more MACs than the in-memory fold reported
    assert (
        abs(chk["macs_gmac"] - stats["mac_after"] / 1e9) <= 0.02 * chk["macs_gmac"]
    ), f"--vae-fold: reloaded MACs {chk['macs_gmac']:.3f} != folded {stats['mac_after']/1e9:.3f} GMAC"
    return out, meta


# ---- INT8 weight compression of the ACTION expert (DEFAULT ON since the 1 Nov promotion) -----
# The action expert is weight-byte-bound on both boxes (219 constant-B MatMuls = 1064.5 of the
# 1117.8 MB bin), so weight-only INT8 (no activation calibration, no data) halves the bytes and the
# GPU plugin folds `Constant u8 -> Convert f16 -> Subtract(zp) -> Multiply(scale) -> MatMul` into its
# `FullyConnectedCompressed` primitive (219 x 13.334 -> 219 x 7.097 ms on the iGPU; layer count
# unchanged on the B580).  Only `DynamicQuantize` is added (0.04 ms/step).
#
# ORDER IS PART OF THE CONTRACT: `split_action.py` parses the **fp16** action graph (layer ids,
# ports, the /Add_5 fan-out), so the B1 split runs FIRST and compression runs on both outputs.
NNCF_VERSION = "3.4.0"


def _nncf_or_die(flag: str = "--action-int8", no_flag: str = "--no-action-int8"):
    """Import nncf, or fail the export loudly with the exact remedy (never silently ship fp16)."""
    import sys as _sys

    try:
        import nncf  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[ov] {flag}: `import nncf` failed in THIS interpreter "
            f"({_sys.executable}): {type(exc).__name__}: {exc}\n"
            "     This INT8 step is ON by default; it is not optional silent behaviour.\n"
            "     Remedy (one of):\n"
            f"       1. {_sys.executable} -m pip install 'nncf=={NNCF_VERSION}'   "
            "   # then re-run this export\n"
            "       2. run the export under a python that already has nncf==3.4.0 + the same "
            "OpenVINO build (the compression is serialised by OvWriter, so mixing OV versions "
            "changes the .bin bytes)\n"
            f"       3. {no_flag}"
            "   (explicitly ship the fp16 graph: no compression)\n"
            "     Reference envs: ptl-test `~/ov_venv_genai` (nncf 3.4.0 + OV master), arl "
            "`~/disk2/ov_venv` (nncf 3.4.0 + OV 2026.4.1), arl `FasterWAM/.venv` (torch export "
            "env, nncf installed 1 Nov 2026)."
        )
    import nncf

    return nncf


def _split_action_lib():
    """Import `split_action.py` (the B1 IR surgery) from wherever it lives on this box."""
    import importlib.util

    here = Path(__file__).resolve().parent
    cands = [
        os.environ.get("ACTION_SPLIT_LIB"),
        str(here / "split_action.py"),
        str(here / "action_b1" / "split_action.py"),
        str(here.parent / "action_b1" / "split_action.py"),
        str(SCRIPTS / "action_b1" / "split_action.py"),
        str(SCRIPTS.parent / "action_b1" / "split_action.py"),
        str(Path.home() / "fasterWAM" / "action_b1" / "split_action.py"),
    ]
    for cand in cands:
        if cand and Path(cand).is_file():
            spec = importlib.util.spec_from_file_location("split_action", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, Path(cand)
    raise SystemExit("[ov] --action-int8: cannot find split_action.py (looked in " f"{[c for c in cands if c]}); pass --no-action-int8 to ship the fp16 graph")


def _const_b_matmuls(model) -> list:
    """`(friendly_name, shape)` of every MatMul whose B input is a 2-D Constant (the weight)."""
    out = []
    for o in model.get_ordered_ops():
        if o.get_type_name() != "MatMul":
            continue
        src = o.inputs()[1].get_source_output()
        if src.get_node().get_type_name() != "Constant":
            continue
        sh = tuple(int(str(d)) for d in src.get_partial_shape())
        if len(sh) == 2:
            out.append((o.get_friendly_name(), sh))
    return out


# Every op NNCF may treat as a weight holder (same list as int8_video/scripts/compress_video.py,
# the script that produced the promoted video bytes: for each of these take the FIRST input that is a
# static `Constant`).  The video profile builds its `ignored_scope` from exactly this enumeration, so
# changing it changes the compressed bytes.
_WEIGHTY_OPS = {
    "MatMul",
    "Gather",
    "Convolution",
    "GroupConvolution",
    "ConvolutionBackpropData",
    "GroupConvolutionBackpropData",
    "MatMulInteger",
    "FullyConnected",
    "EmbeddingBagOffsetsSum",
}

# ---- weight-only INT8 profiles ----------------------------------------------------------------
# One recipe for both units (`INT8_SYM`, ratio 1.0, group_size -1, subset_size 1, data-free).  They
# differ in (a) whether the non-target weight holders must be named in NNCF's `ignored_scope`, (b) the
# node counts that prove the declared scope, (c) the OPT-IN archive directory name.  Nothing is
# archived by default: the fp16 bytes are parked as `<stem>.fp16tmp.*` and deleted once the INT8
# graph verifies.  `FWAM_KEEP_FP16_ARCHIVE=1` keeps them (in (c)) as the pre-promotion control.
#   ACTION: NNCF leaves action's handful of non-2-D constants alone by itself, so `ignored_scope`
#           stays None -- that is what the promoted 1 Nov bytes were built with.  Do not "improve" it.
#   VIDEO : 435 weight-bearing nodes, 287 of them MatMul targets and 148 (147 Gather tables + 1 Conv)
#           that NNCF WOULD compress if they were not named explicitly.
PROFILE_ACTION = {
    "name": "action",
    "flag": "--action-int8",
    "no_flag": "--no-action-int8",
    "archive": "ir_fp16",
    "ignore_by_name": False,
    "expect_targets": None,  # caller-dependent: action_fp16 219, action_b1 201
    "expect_matmuls": None,
    "expect_ignored": None,
}
PROFILE_VIDEO = {
    "name": "video",
    "flag": "--video-int8",
    "no_flag": "--no-video-int8",
    "archive": "ir_pre_video_int8",
    "ignore_by_name": True,
    # 228 x 3072^2 + 28 x 3072->14336 + 28 x 14336->3072 + 3072->18432 + 4096->3072 + 256->3072
    "expect_targets": 287,
    "expect_matmuls": 399,  # the 112 dynamic-B (attention / q.k^T) MatMuls stay MatMuls
    "expect_ignored": 148,  # 147 Gather tables + 1 Convolution, must stay fp16
}


def _profile_expect(profile: dict) -> tuple:
    """`(targets, matmuls, ignored)` expected for this profile; overridable for a fresh config.

    `VIDEO_INT8_EXPECT="287,399,148"` exists so an intentional `--unit video` export with a different
    flag set (e.g. `--rope3 0`) can still ship; it prints what it overrode.
    """
    exp = (profile["expect_targets"], profile["expect_matmuls"], profile["expect_ignored"])
    raw = os.environ.get("VIDEO_INT8_EXPECT")
    if raw and profile["name"] == "video":
        exp = tuple(int(x) for x in raw.split(","))
        print(f"[ov] {profile['flag']}: VIDEO_INT8_EXPECT override -> targets/matmuls/ignored={exp}", flush=True)
    return exp


def _weight_nodes(model) -> list:
    """`(name, type, const_port, shape, element_type)` for every weight-bearing node.

    Identical enumeration to `int8_video/scripts/compress_video.py`; used to (a) build the video
    `ignored_scope` by name and (b) prove afterwards that those weights were left alone.
    """
    out = []
    for o in model.get_ordered_ops():
        tn = o.get_type_name()
        if tn not in _WEIGHTY_OPS:
            continue
        for pi, inp in enumerate(o.inputs()):
            src = inp.get_source_output()
            if src.get_node().get_type_name() != "Constant":
                continue
            ps = src.get_partial_shape()
            if not ps.is_static:
                continue
            out.append((o.get_friendly_name(), tn, pi, tuple(int(str(d)) for d in ps), src.get_element_type()))
            break
    return out


def _keep_fp16_archive() -> bool:
    """True when the caller asked for the pre-promotion fp16 IRs to be KEPT (default: no).

    Set `FWAM_KEEP_FP16_ARCHIVE=1` to archive instead of deleting.  Nothing else changes: the same
    env var covers the action and the video unit.
    """
    return os.environ.get("FWAM_KEEP_FP16_ARCHIVE", "").strip().lower() not in ("", "0", "false", "no", "off")


def _compress_action_ir(xml: Path, archive_dir: Path | None, compress: bool, nncf, profile: dict = PROFILE_ACTION) -> dict:
    """Weight-only INT8 one IR in place, dropping the fp16 bytes unless an archive is asked for.

    The fp16 pair is parked by RENAME before anything is written -- into `archive_dir` when one is
    given, otherwise as `<stem>.fp16tmp.{xml,bin}` in the SAME directory.  A rename inside one
    directory costs no disk, and the mmap of the model that was already loaded stays valid either
    way.  With no archive (`archive_dir=None`) the parked bytes are an intermediate: they are
    unlinked as soon as the INT8 graph has been written, and restored if compression/save fails.

    Returns stats (`"archive": None` when the fp16 bytes were dropped).  Asserts afterwards that
    (a) the bin shrank to ~half, (b) every constant-B MatMul weight is now a dequant chain (no
    MatMul keeps a 2-D `Constant` B -- that is the one the plugin would NOT fuse), (c) the reloaded
    graph still has all its MatMuls, and (d, video profile only) the 148 explicitly ignored
    Gather/Conv weight nodes are still uncompressed (no u8/i8 weight).  NB a failure in those
    POST-write asserts leaves the fp16 pair as `<stem>.fp16tmp.*` next to the IR (the error is
    raised, so nothing pretends otherwise); `rm */*.fp16tmp.*` then re-export.
    """
    import openvino as ov

    xml = Path(xml)
    flag = profile["flag"]
    bin_path = xml.with_suffix(".bin")
    if archive_dir is not None:
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
    bin_before = bin_path.stat().st_size
    dur = time.perf_counter()
    core = ov.Core()
    model = core.read_model(str(xml))  # mmap of the .bin we are about to move
    read_s = time.perf_counter() - dur
    cands = _const_b_matmuls(model)
    assert cands, f"{xml.name}: no constant-B MatMul found -- is this really a " f"{profile['name']} IR?"
    n_mm_before = sum(1 for o in model.get_ordered_ops() if o.get_type_name() == "MatMul")
    # ---- the declared scope, checked BEFORE anything is written --------------------------
    exp_targets, exp_matmuls, exp_ignored = _profile_expect(profile)
    ignored = []
    if profile["ignore_by_name"]:
        target_names = {n for n, _ in cands}
        wn = _weight_nodes(model)
        ignored = [n for n, tn, pi, sh, et in wn if n not in target_names]
        print(f"[ov] {flag}: {len(wn)} weight-bearing nodes -> {len(cands)} targets, " f"{len(ignored)} explicitly ignored (Gather/Conv/non-2-D)", flush=True)
        if exp_targets is not None:
            assert len(cands) == exp_targets, (
                f"{xml.name}: {len(cands)} constant-B MatMuls, expected {exp_targets} for the "
                f"{profile['name']} profile (this is the promoted scope; set "
                f'VIDEO_INT8_EXPECT="targets,matmuls,ignored" if this export really is a '
                "different graph)"
            )
        if exp_ignored is not None:
            assert len(ignored) == exp_ignored, (
                f"{xml.name}: {len(ignored)} weight nodes would be ignored, expected "
                f"{exp_ignored} (147 Gather tables + 1 Convolution) -- NNCF would compress the "
                "rest, so the scope has to be re-declared, not guessed"
            )
    if exp_matmuls is not None:
        assert n_mm_before == exp_matmuls, f"{xml.name}: {n_mm_before} MatMuls, expected {exp_matmuls} for the " f"{profile['name']} profile"
    # park the fp16 bytes by RENAME: into the archive when one was asked for, else beside itself
    keep_fp16 = archive_dir is not None
    park_dir = archive_dir if keep_fp16 else xml.parent
    park_xml = park_dir / (xml.name if keep_fp16 else f"{xml.stem}.fp16tmp.xml")
    park_bin = park_dir / (bin_path.name if keep_fp16 else f"{xml.stem}.fp16tmp.bin")
    os.replace(str(xml), str(park_xml))
    os.replace(str(bin_path), str(park_bin))
    # compress the already-loaded in-memory model and write the SAME filenames back
    tmp_xml = xml.with_name(xml.stem + ".i8tmp.xml")
    try:
        t0 = time.perf_counter()
        cm = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT8_SYM,
            ratio=1.0,
            group_size=-1,
            ignored_scope=nncf.IgnoredScope(names=ignored) if ignored else None,
            all_layers=None,
            subset_size=1,
        )
        comp_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ov.save_model(cm, str(tmp_xml), compress_to_fp16=bool(compress))
        save_s = time.perf_counter() - t0
        os.replace(str(tmp_xml), str(xml))
        os.replace(str(tmp_xml.with_suffix(".bin")), str(bin_path))
        del cm, model
    except BaseException:
        # rollback: drop the half-written INT8 bytes and put the fp16 pair back where it was
        for p in (xml, bin_path, tmp_xml, tmp_xml.with_suffix(".bin")):
            if p.exists():
                p.unlink()
        os.replace(str(park_xml), str(xml))
        os.replace(str(park_bin), str(bin_path))
        raise
    # ---- verification of the bytes that are now in place
    # After compression a MatMul's B input is no longer a Constant but
    # `Convert(u8 weights) -> Subtract(zp) -> Multiply(scale) -> MatMul`, so count the u8 WEIGHT
    # constants instead (one [K,N] u8/i8 constant per compressed MatMul).  Both axes must be > 1:
    # the per-row/per-channel zero-points are [1,N] (action unit) or [K,1] (video unit), and the
    # scales are float.  NB: str(element_type) spells `<Type: 'uint8_t'>` on the 2026.5 build and
    # `<Type: uint8>` elsewhere -- compare via ov.Type, never by string.
    om = core.read_model(str(xml))  # the bytes that are now on disk
    u8t = (ov.Type.u8, ov.Type.i8)
    n_u8 = 0
    for o in om.get_ordered_ops():
        if o.get_type_name() != "Constant" or o.get_output_element_type(0) not in u8t:
            continue
        sh = tuple(int(str(d)) for d in o.get_output_partial_shape(0))
        if len(sh) == 2 and sh[0] > 1 and sh[1] > 1:
            n_u8 += 1
    n_mm = sum(1 for o in om.get_ordered_ops() if o.get_type_name() == "MatMul")
    assert n_mm == n_mm_before, f"{xml.name}: MatMul count {n_mm_before} -> {n_mm}"
    # what the GPU plugin matches: B is no longer a Constant but the tail of a dequant chain.
    # NB this count is a FLOOR, not the target count: on the video unit it reads 343 = 287
    # compressed + 56 MatMuls whose B was already a `Multiply` (the rope3 basis folded into the
    # q/k weights), so assert `>=` targets, never `==`.
    n_dq = 0
    for o in om.get_ordered_ops():
        if o.get_type_name() != "MatMul":
            continue
        if o.inputs()[1].get_source_output().get_node().get_type_name() in ("Multiply", "Convert", "Subtract"):
            n_dq += 1
    bin_after = bin_path.stat().st_size
    ratio = bin_after / max(bin_before, 1)
    n_cb_after = len(_const_b_matmuls(om))
    assert n_cb_after == 0, (
        f"{xml.name}: {n_cb_after} MatMuls still have a Constant B (all {len(cands)} targets must be "
        "dequant chains; a surviving Constant weight is one the plugin will NOT fuse into "
        "FullyConnectedCompressed)"
    )
    assert n_dq >= len(cands), f"{xml.name}: {n_dq} MatMuls have a dequant-chain B input, expected " f">= {len(cands)}"
    assert len(cands) <= n_u8 <= len(cands) + 16, (
        f"{xml.name}: {n_u8} u8/i8 weight constants for {len(cands)} target MatMuls " f"({flag} expects ~one per compressed MatMul plus a few NNCF extras)"
    )
    # ---- (d) the EXPLICITLY IGNORED weight holders must be untouched ---------------------
    # After compression a target MatMul has no Constant input left, so `_weight_nodes` now returns
    # exactly the ignored ones: for the video unit 147 int64 `Gather` index/frequency tables + 1
    # fp16 `Convolution` filter.  They must still be uncompressed -- a quantised Gather table or
    # Conv filter would be out-of-scope weight loss the latency measurement could not explain.
    n_kept, kept_types = 0, {}
    if profile["ignore_by_name"]:
        wn_after = _weight_nodes(om)
        n_kept = len(wn_after)
        bad = [(n, tn, str(et)) for n, tn, pi, sh, et in wn_after if et in u8t]
        assert not bad, (
            f"{xml.name}: {len(bad)} ignored weight node(s) were compressed anyway " f"({[b[0] for b in bad[:4]]}) -- the ignored_scope did not take"
        )
        for n, tn, pi, sh, et in wn_after:
            kept_types[str(et)] = kept_types.get(str(et), 0) + 1
        if exp_ignored is not None:
            assert n_kept == exp_ignored, (
                f"{xml.name}: {n_kept} weight-bearing nodes left with a Constant weight, expected "
                f"{exp_ignored}; the other {len(cands)} MatMul weights are u8 dequant chains now"
            )
        print(f"[ov] {flag}: {n_kept} ignored weight nodes left uncompressed {kept_types}", flush=True)
    assert 0.40 <= ratio <= 0.62, f"{xml.name}: bin {bin_before/1e6:.1f} -> {bin_after/1e6:.1f} MB " f"(ratio {ratio:.3f}) is not the expected ~0.50"
    if not keep_fp16:
        # the parked fp16 bytes are an intermediate: the INT8 graph is written and verified
        park_xml.unlink()
        park_bin.unlink()
        park_xml = None
    return {
        "xml": str(xml),
        "bin_mb_before": round(bin_before / 1e6, 1),
        "bin_mb_after": round(bin_after / 1e6, 1),
        "ratio": round(ratio, 4),
        "matmuls": len(cands),
        "i8_weights": n_u8,
        "dequant_b": n_dq,
        "matmuls_before": n_mm_before,
        "matmuls_after": n_mm,
        "ignored_nodes": len(ignored),
        "kept_weights": n_kept,
        "kept_types": kept_types,
        "archive": str(park_xml) if park_xml else None,
        "read_s": round(read_s, 2),
        "compress_s": round(comp_s, 2),
        "save_s": round(save_s, 2),
    }


def apply_action_int8(info: dict, compress: bool = True, out_root: Path | None = None, archive_dir: Path | None = None) -> tuple:
    """B1 split FIRST, then INT8 both action graphs, in place, same filenames.

    `info` is the `to_ir` dict of the freshly exported action unit (fp16).  Writes
      <unit>/action_fp16.{xml,bin}            -- INT8 (replaces the fp16 bytes)
      <unit>/action_b1/action_b1.{xml,bin}    -- INT8
      <unit>/action_b1/action_ctx.{xml,bin}   -- fp16 (outside the loop, already hidden)

    NO fp16 copy is kept: each fp16 pair is parked as `<stem>.fp16tmp.*` beside itself (a rename,
    no extra disk) and deleted once the INT8 graph has been written and verified -- those bytes are
    an intermediate of the conversion, and the fp16 arm is reproducible with `--no-action-int8`.
    `FWAM_KEEP_FP16_ARCHIVE=1` (or an explicit `archive_dir`) keeps them instead, in
    `<archive_dir>` = `<out-dir>/ir_fp16` by default, as the pre-promotion control.
    """
    import subprocess

    nncf = _nncf_or_die()
    xml = Path(info["xml"])
    ir_dir = xml.parent
    b1_dir = ir_dir / "action_b1"
    if archive_dir is None and _keep_fp16_archive():
        archive_dir = (Path(out_root) / "ir_fp16") if out_root else (ir_dir.parent / "ir_fp16")
    t0 = time.perf_counter()
    # ---- (1) B1 split -- MUST be first: it parses the fp16 graph (ids/ports/fan-out)
    lib, lib_path = _split_action_lib()
    src_xml = ir_dir / xml.name
    # split_action.py hardcodes `action_fp16.{xml,bin}` inside `--src`; when the exporter named the
    # graph differently (`--dtype fp32` -> action_fp32.xml) stage symlinks under that canonical name
    # so the surgery always reads the graph we just built.
    split_src = ir_dir
    if xml.name != "action_fp16.xml":
        split_src = ir_dir / "_split_src"
        split_src.mkdir(exist_ok=True)
        for ext in (".xml", ".bin"):
            link = split_src / f"action_fp16{ext}"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(Path(xml.with_suffix(ext)).resolve(), link)
    split = [sys.executable, str(lib_path), "--src", str(split_src), "--out", str(b1_dir)]
    print(f"[ov] --action-int8: splitting the FP16 graph first ({src_xml.name} -> {b1_dir})", flush=True)
    proc = subprocess.run(split, capture_output=True, text=True)
    if split_src != ir_dir:
        shutil.rmtree(split_src, ignore_errors=True)  # staging symlinks, not an artifact
    if proc.returncode != 0:
        raise SystemExit(
            f"[ov] --action-int8: split_action.py failed (rc={proc.returncode})\n"
            f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}\n"
            "     Reminder: the split needs an FP16 action IR (it asserts FP16 on the tower "
            "terminals).  Export with `--trace-half` -- that is the mode the promoted "
            "`ir/int8/action/action_fp16.*` bytes came from.  `--dtype fp16` alone only names the "
            "file: the ONNX is still traced from fp32 torch, so intermediate ports stay FP32."
        )
    split_s = time.perf_counter() - t0
    b1_xml = b1_dir / "action_b1.xml"
    ctx_xml = b1_dir / "action_ctx.xml"
    for p in (b1_xml, ctx_xml, b1_xml.with_suffix(".bin"), ctx_xml.with_suffix(".bin")):
        assert p.exists(), f"--action-int8: split_action.py did not produce {p}"
    # ---- (2) compress both graphs in place (fp16 bytes parked, then dropped unless archived)
    st_src = _compress_action_ir(src_xml, archive_dir, compress, nncf)
    st_b1 = _compress_action_ir(b1_xml, (archive_dir / "action_b1") if archive_dir else None, compress, nncf)
    meta = {
        "nncf": getattr(nncf, "__version__", "?"),
        "split_lib": str(lib_path),
        "split_s": round(split_s, 2),
        "action_fp16": st_src,
        "action_b1": st_b1,
        "action_ctx_fp16_mb": round(ctx_xml.with_suffix(".bin").stat().st_size / 1e6, 1),
        "archive_dir": str(archive_dir) if archive_dir else None,
    }
    out = dict(info, model=None, bin_mb=st_src["bin_mb_after"], n_ops=None, action_int8=meta, xml=xml)
    return out, meta


def apply_video_int8(info: dict, compress: bool = True, out_root: Path | None = None, archive_dir: Path | None = None) -> tuple:
    """INT8-compress the video prefill graph in place, same filenames, no other step.

    ORDER CONTRACT: the video unit has no split (unlike the action unit, where `split_action.py`
    parses the fp16 graph and must run BEFORE compression), so this is simply
    export -> compress -> `os.replace`.  Writes
      <unit>/video_fp16.{xml,bin}          -- INT8 (replaces the fp16 bytes)

    No fp16 copy is kept: the fp16 pair is parked as `video_fp16.fp16tmp.*` beside itself (a
    rename, no extra disk) and deleted once the INT8 graph verifies -- it is an intermediate of the
    conversion, and the fp16 arm is reproducible with `--no-video-int8`.  `FWAM_KEEP_FP16_ARCHIVE=1`
    or an explicit `archive_dir` / `--video-int8-archive` keeps it instead, in `<archive_dir>` =
    `<out-dir>/../ir_pre_video_int8`.  NOTE (4 Nov): `$T/ir` is a symlink into `$T/ir_sets/<set>/ir`,
    and the path is RESOLVED before the two `parent`s, so for `--out-dir $T/ir` the archive lands at
    `$T/ir_sets/<active set>/ir_pre_video_int8/` -- i.e. a compression can never overwrite the fp16
    master of another set.
    """
    nncf = _nncf_or_die(PROFILE_VIDEO["flag"], PROFILE_VIDEO["no_flag"])
    xml = Path(info["xml"])
    if archive_dir is None and _keep_fp16_archive():
        if out_root is None:
            raise SystemExit(
                "[ov] --video-int8: FWAM_KEEP_FP16_ARCHIVE=1 but no --out-dir to " f"place the fp16 archive of {xml} (or pass --video-int8-archive)"
            )
        archive_dir = Path(out_root).resolve().parent.parent / PROFILE_VIDEO["archive"]
    t0 = time.perf_counter()
    st = _compress_action_ir(xml, archive_dir, compress, nncf, PROFILE_VIDEO)
    meta = {
        "nncf": getattr(nncf, "__version__", "?"),
        "video_fp16": st,
        "archive_dir": str(archive_dir) if archive_dir else None,
        "total_s": round(time.perf_counter() - t0, 2),
    }
    out = dict(info, model=None, bin_mb=st["bin_mb_after"], n_ops=None, video_int8=meta, xml=xml)
    return out, meta


def bench_device(info: dict, device: str, ordered: list, warmup: int = 5, runs: int = 20) -> dict:
    core, om = info["core"], info["model"]
    try:
        t0 = time.perf_counter()
        cm = core.compile_model(om, device)
        compile_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return {"device": device, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    if len(cm.inputs) != len(ordered):
        return {"device": device, "error": f"input count {len(cm.inputs)} != {len(ordered)}"}
    try:
        for _ in range(warmup):
            cm(ordered)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            cm(ordered)
            ts.append((time.perf_counter() - t0) * 1e3)
        return {
            "device": device,
            "mean_ms": float(np.mean(ts)),
            "min_ms": float(np.min(ts)),
            "p90_ms": float(np.percentile(ts, 90)),
            "compile_s": round(compile_s, 2),
            "rss_peak_mb": rss_mb()[1],
        }
    except Exception as exc:  # noqa: BLE001
        return {"device": device, "compile_s": round(compile_s, 2), "rss_peak_mb": rss_mb()[1], "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def infer_on(info: dict, device: str, ordered: list) -> torch.Tensor:
    cm = info["core"].compile_model(info["model"], device)
    out = cm(ordered)
    # NB: in this source build CompiledModel.output is a METHOD; OVDict indexes by int/Output
    return torch.from_numpy(np.asarray(out[0]).copy()).float()


# ------------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--unit", required=True, choices=["action", "vae", "video", "video_slice"])
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"], help="IR precision (ONNX is always traced from fp32 torch; fp16 = compress_to_fp16)")
    p.add_argument("--out-dir", default="/home/clee30/disk2/models/fasterWAM/ir")
    p.add_argument("--model-dir", default=DEFAULT_CKPT, help="checkpoint .pt (hustvl/FasterWAM libero)")
    p.add_argument(
        "--task", default=None, help="hydra task override, e.g. robotwin_fasterwam_3cam_384_1e-4 (default: " "measure_latency's model->LIBERO task mapping)"
    )
    p.add_argument(
        "--art-suffix",
        default="",
        help="suffix for scripts/artifacts/ov_<unit><suffix>_ref.pt + the log name, so a "
        "second task's refs/logs do not clobber the first task's (e.g. _robotwin)",
    )
    p.add_argument("--ref", default=DEFAULT_REF, help="dumped smoke-test inputs")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--devices", default="CPU,GPU.0,GPU.1")
    p.add_argument("--dry-run", action="store_true", help="wrapper in torch only, no ONNX")
    p.add_argument("--sin64", type=int, default=1, help="1 = fp64 time-embedding (matches repo), 0 = fp32")
    p.add_argument(
        "--fp64-probe",
        action="store_true",
        help="also run the wrapper in float64 and compare, to separate fp32 round-off "
        "from real graph error (needed when max_rel is dominated by near-zero elements)",
    )
    p.add_argument(
        "--fp16-io",
        dest="fp16_io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fp16 IRs declare fp16 bulk inputs/outputs (explicit Convert nodes; see set_fp16_interface). "
        "--no-fp16-io = the pre-14-Nov f32 interface. No effect on --dtype fp32.",
    )
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--layers", type=int, default=3, help="--unit video_slice: number of leading video-expert blocks to export")
    p.add_argument(
        "--rope2",
        type=int,
        default=1,
        help="1 (default since the 29 Oct promotion) = pair-swap-free RoPE (3 kernels per "
        "application instead of 9, bit-identical); 0 = the pre-promotion shipping form",
    )
    p.add_argument(
        "--rope3",
        type=int,
        default=1,
        help="1 (default since the 30 Oct promotion) = RoPE in the half-major (GPT-NeOX) basis "
        "so OpenVINO fuses it to ONE "
        "ocl::rope::opt kernel.  Takes precedence over "
        "--rope2 and additionally RE-BASES the q/k weights of every self-attention "
        "(apply_rope3_basis), so the exported video K/V interface is in that basis "
        "too: a rope3 video IR and a rope3 action IR must be used together.",
    )
    p.add_argument(
        "--tmod2",
        type=int,
        default=1,
        help="1 (default since the 29 Oct promotion) = video t_mod from its 2 distinct token "
        "timesteps (M=2) instead of all 294; 0 = pre-promotion form.  Exact by "
        "construction, see pre_dit_tmod2",
    )
    p.add_argument(
        "--modg",
        type=int,
        default=1,
        help="1 (default since the 29 Oct promotion) = keep video t_mod as ONE ROW PER "
        "LATENT FRAME (needs --tmod2 1) and broadcast it against tokens reshaped to "
        "(B,f,tpf,C); deletes the 28 per-block (modulation + t_mod) adds (-7.8 ms/chunk); "
        "0 = pre-promotion per-token form.  Auditable: with 1 the IR has no 1x294x6x3072 op",
    )
    p.add_argument(
        "--txtp",
        type=int,
        default=0,
        help="text-context projection layout variant 0..8 (0 = untouched shipping graph, " "see _TxtProj / TXTP_MODES); experimental, default off",
    )
    p.add_argument(
        "--ctxpad",
        type=int,
        default=0,
        help="pad the text/context stream to a multiple of N inside pre_dit and extend the "
        "cross-attn mask (-1e4) for the pad keys, so every odd-M text GEMM becomes "
        "M=ceil(L/8)*8=136 (1 -> the default alignment 8; 0 = untouched shipping graph, "
        "see _ctxpad_pair / _pre_dit_ctxpad).  Applies to BOTH experts (video prefill "
        "and the action unit's own text_embedding); experimental, default off",
    )
    p.add_argument(
        "--vae-fold",
        dest="vae_fold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--unit vae only: fold the encoder's 22 dead depth taps (Conv3D kd=3 on a D=1 "
        "tensor behind a depth-only Pad -> Conv3D kd=1, weights sliced at the live depth "
        "index) right after the ONNX->OV conversion.  233.0 -> 81.4 GMAC, 296 -> 107.8 MB "
        "bin, -58 %% VAE stage; the promoted `ir/vae` bytes since 31 Oct.  The transform "
        "is imported from vae_opt/fold_vae.py (this flag is the only wiring).  "
        "--no-vae-fold reproduces the pre-promotion (unfolded) graph",
    )
    p.add_argument(
        "--action-int8",
        dest="action_int8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--unit action only: after the ONNX->OV conversion, run the B1 split "
        "(action_b1/action_ctx) and NNCF weight-only INT8-compress the 219 "
        "constant-B MatMuls of action_fp16 AND the 201 of action_b1, replacing the "
        "fp16 bytes under the same filenames.  NO fp16 copy is kept: the fp16 bytes "
        "of each graph are parked as `*.fp16tmp.*` beside it (a rename, no extra "
        "disk) and deleted once the INT8 graph verifies -- set "
        "FWAM_KEEP_FP16_ARCHIVE=1 to keep them in <out-dir>/action/ir_fp16/ instead.  "
        "Bins halve; the GPU plugin fuses the dequant "
        "into FullyConnectedCompressed: -48.7 ms/chunk iGPU, -7.1 ms B580.  Shipped "
        "default since the 1 Nov promotion; --no-action-int8 reproduces the fp16 "
        "pre-promotion graph (and writes no action_b1).  Needs nncf==3.4.0 in THIS "
        "interpreter (the script fails loudly, with the remedy, if it is missing)",
    )
    p.add_argument(
        "--video-int8",
        dest="video_int8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--unit video ONLY (never video_slice / action / vae): after the ONNX->OV "
        "conversion, NNCF weight-only INT8-compress the 287 constant-B MatMuls of "
        "the 5 B video prefill (the 148 Gather/Conv weight nodes are named in the "
        "ignored_scope and stay fp16) and replace the fp16 bytes under the same "
        "filenames; the fp16 original is dropped the same way (parked as "
        "`video_fp16.fp16tmp.*`, unlinked once the INT8 graph verifies -- set "
        "FWAM_KEEP_FP16_ARCHIVE=1 or --video-int8-archive to keep it).  9.382 -> "
        "4.697 GB; the GPU plugin folds the dequant into "
        "FullyConnectedCompressed: video prefill -50.1 ms iGPU, -16.9 ms B580 per "
        "chunk.  Shipped default since the 1 Nov promotion; --no-video-int8 "
        "reproduces the fp16 graph (the rollback path).  No split step here, i.e. "
        "export -> compress (the action unit is export -> split -> compress).  "
        "Needs nncf==3.4.0 in THIS interpreter (the script fails loudly, with the "
        "remedy, if it is missing)",
    )
    p.add_argument(
        "--video-int8-archive",
        default=None,
        help="opt-in override: where --video-int8 KEEPS the fp16 video IR it replaces "
        "(default: keep nothing.  With FWAM_KEEP_FP16_ARCHIVE=1 the default is "
        "<out-dir>/../ir_pre_video_int8 -- with --out-dir $T/ir (a symlink into "
        "ir_sets/<set>/ir) that resolves to $T/ir_sets/<set>/ir_pre_video_int8)",
    )
    p.add_argument(
        "--trace-half",
        action="store_true",
        help="trace the graph with HALF-precision weights/activations instead of fp32. "
        "For the 5 B video prefill this halves ONNX+IR size (20 GB -> 10 GB) and "
        "RAM; the resulting IR is already fp16, so compress_to_fp16 is skipped. "
        "Matches what the GPU plugin does internally to an fp32 IR anyway.",
    )
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--reuse-ref", action="store_true", help="reuse scripts/artifacts/ov_<unit>_ref.pt instead of re-deriving the torch reference")
    args = p.parse_args()

    torch.set_num_threads(int(args.threads))
    torch.set_grad_enabled(False)
    import warnings

    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    tag = args.unit if args.unit != "video_slice" else f"video_slice{args.layers}"
    out_root = Path(args.out_dir) / tag
    art = SCRIPTS / "artifacts"
    art.mkdir(exist_ok=True, parents=True)
    fp16 = (args.dtype == "fp16") or bool(args.trace_half)
    compress = not bool(args.trace_half)  # fp16-weight ONNX needs no fp16 compression
    prec = "fp16w" if args.trace_half else args.dtype
    log = {
        "unit": args.unit,
        "layers": args.layers if args.unit == "video_slice" else None,
        "dtype": args.dtype,
        "trace_half": int(args.trace_half),
        "prec": prec,
        "rope2": int(args.rope2),
        "rope3": int(args.rope3),
        "ts": time.strftime("%F %T"),
    }

    def emit(k, v):
        log[k] = v
        print(f"[ov] {k}: {v}", flush=True)

    # ---- reference (torch, UNPATCHED repo path) ----------------------------------
    ref_path = art / f"ov_{args.unit}{args.art_suffix}_ref.pt"
    if args.reuse_ref and ref_path.exists():
        ref = torch.load(str(ref_path), map_location="cpu", weights_only=False)
        emit("ref_source", f"cached {ref_path}")
    t0 = time.perf_counter()
    model, spec, facts = build(args)
    emit("build_s", round(time.perf_counter() - t0, 1))
    emit("rss_after_build_mb", rss_mb())
    emit("build_facts", {k: facts[k] for k in ("n_missing", "n_unexpected", "payload_torch_dtype")})
    if not (args.reuse_ref and ref_path.exists()):
        dump = torch.load(args.ref, map_location="cpu", weights_only=False)
        inputs = dump["inputs"]
        emit("inputs", {k: (list(v.shape), str(v.dtype)) for k, v in inputs.items()})
        with torch.no_grad():
            if args.unit == "action":
                ref = derive_action_ref(model, spec, inputs)
            elif args.unit == "vae":
                ref = derive_vae_ref(model, inputs)
            elif args.unit == "video_slice":
                ref = derive_video_ref(model, spec, inputs, keep_blocks=8)
            else:
                ref = derive_video_ref(model, spec, inputs, keep_blocks=8)
        torch.save(ref, str(ref_path))
        emit("ref_saved", str(ref_path))
    if args.unit in ("video_slice", "video"):
        emit("ref_out", [[list(t.shape), float(t.sum())] for t in (ref["kv_k"][0], ref["kv_v"][-1])])
    else:
        emit("ref_out", [list(ref.get("pred", ref.get("z")).shape), float(ref.get("pred", ref.get("z")).sum())])

    if args.unit == "video":
        print("[ov] unit=video: full 30-block prefill (5 B).", flush=True)

    # ---- build wrapper (patched rope/masks), check it against the unpatched ref ---
    patch_all(
        sin64=bool(args.sin64),
        rope2=bool(args.rope2),
        tmod2=bool(args.tmod2),
        modg=bool(args.modg),
        txtp=int(args.txtp),
        ctxpad=int(args.ctxpad),
        rope3=bool(args.rope3),
    )
    rebuild_freqs(model)
    if int(args.rope3):
        # weights (not just the rope function) must move to the half-major basis; the reference above
        # was derived on the UNPATCHED path, so it stays in the pair-major basis by construction
        emit("rope3_basis", apply_rope3_basis(model))
    ref_outputs = None
    if args.unit == "action":
        cond = ref["cond_layers"]
        wrapper = ActionStep(model, ref["masks"], ref["ctx_mask"], ref["video_seq_len"], cond)
        names = {"in": ["latents_action", "t", "context"] + [f"kv{i}_{c}" for i in range(len(cond)) for c in ("k", "v")], "out": ["pred"]}
        # `--rope3`: the reference K/V were produced by the UNPATCHED (pair-major) path, but a rope3
        # action graph expects them in the half-major basis -- P is part of the interface, so re-base
        # the inputs before feeding the wrapper (V is not part of the rotation, so it stays).
        kvk = [rope3_permute(t, ROPE3["head_dim"]) for t in ref["kv_k"]] if int(args.rope3) else list(ref["kv_k"])
        if int(args.rope3):
            emit("rope3_kv_rebased", [list(t.shape) for t in kvk[:1]])
        sample = (ref["latents_action"], ref["t"], ref["context"], *[t for pair in zip(kvk, ref["kv_v"]) for t in pair])
        reference = ref["pred"]
    elif args.unit in ("video_slice", "video"):
        full = args.unit == "video"
        n = None if full else int(args.layers)
        cond_all = ref["cond_layers"]
        cond = [i for i in cond_all if full or i < n]
        wrapper = VideoPrefill(model, ref["vmask"], layers=n, out_hidden=not full)
        names = {
            "in": ["latents", "timestep", "context", "context_mask"],
            "out": [f"kv{i}_{c}" for i in cond for c in ("k", "v")] + ([] if full else ["hidden"]),
        }
        sample = (ref["latents"], ref["t"], ref["context"], ref["ctx_mask"])
        reference = None
        ref_outputs = [t for pair in zip([ref["kv_k"][cond_all.index(i)] for i in cond], [ref["kv_v"][cond_all.index(i)] for i in cond]) for t in pair]
        if not full:
            ref_outputs = ref_outputs + [ref["blk_out"][n - 1]]
        emit("slice_cond_layers", cond)
        emit("slice_inputs", {k: list(v.shape) for k, v in zip(names["in"], sample)})
    else:
        wrapper = VaeEncode1(model.vae)
        names = {"in": ["image"], "out": ["latent"]}
        sample = (ref["x"],)
        reference = ref["z"]
    wrapper.eval()
    if args.unit == "action":
        vs = ref["video_seq_len"]
        arow = [bool(ref["masks"][l][vs : vs + 32, : vs + 32].all()) for l in ref["cond_layers"]]
        emit("action_row_masks_all_true", arow)
        emit("ctx_mask_all_true", bool(ref["ctx_mask"].all()))

    n_kv_out = 2 * len(cond) if args.unit in ("video_slice", "video") else 0

    def rope3_cmp(i, g):
        """`--rope3`: a video K leaves the graph in the HALF-MAJOR basis; compare in the ref's basis.

        Every gate that compares a K output against the unpatched reference has to undo P first -- the
        torch fp32 pass, the fp16-trace pass and `--verify` on the compiled IR.  Skipping one makes a
        perfectly good rope3 graph report cos ~0.03-0.09 on its K outputs (V never needs it).
        """
        if int(args.rope3) and i < n_kv_out and i % 2 == 0:
            return rope3_permute(g, ROPE3["head_dim"], inverse=True)
        return g

    with torch.no_grad():
        got = wrapper(*sample)
    if int(args.rope3):
        # every attention the traced graph actually contains (see _rope3_group_check)
        emit("rope3_attn_group_sizes", sorted(ROPE3.get("attn_group_sizes", set())))
    if ref_outputs is not None:
        outs = list(got) if isinstance(got, tuple) else [got]
        worst = {"max_abs": 0.0, "max_rel": 0.0, "rel_to_range": 0.0}
        for i, (r, g) in enumerate(zip(ref_outputs, outs)):
            g = rope3_cmp(i, g)
            m = metrics(r, g)
            emit(f"torch_slice_out{i}_vs_ref", fmt(m))
            for kk in worst:
                worst[kk] = max(worst[kk], m[kk])
        m_patch = {"max_abs": worst["max_abs"], "max_rel": worst["max_rel"], "rmse": 0.0, "cos": 1.0, "rel_to_range": worst["rel_to_range"]}
        reference = ref_outputs[-1]
        got = outs[-1]
        emit("torch_slice_worst", fmt(m_patch))
    else:
        m_patch = metrics(reference, got)
        emit("torch_patched_wrapper_vs_ref", fmt(m_patch))
    print(f"[ov] ref sum={float(reference.sum()):.6f} wrapped sum={float(got.sum()):.6f} " f"shape={list(got.shape)}", flush=True)
    n_params = sum(int(x.numel()) for x in wrapper.parameters())
    emit("wrapper_params", n_params)

    # ---- optional half-precision trace ------------------------------------------
    # Tracing with fp16 weights halves ONNX/IR size (20 GB -> 10 GB for the 5 B prefill) and the
    # RAM needed to read/convert it.  Numerically it is close to what the OV GPU plugin does to an
    # fp32 IR by default anyway (it compresses fp32 weights to fp16 and runs fp16 math).
    inp_dtype = torch.float32
    if args.trace_half:
        if args.unit in ("video_slice", "video"):
            model.video_expert.half()
        elif args.unit == "action":
            model.action_expert.half()
        wrapper = wrapper.half()  # wrapper-owned mask buffers -> fp16
        sample = tuple(t.half() if t.is_floating_point() else t for t in sample)
        inp_dtype = torch.float16
        emit("rss_after_half_mb", rss_mb())
        with torch.no_grad():
            got_h = wrapper(*sample)
        outs_h = list(got_h) if isinstance(got_h, tuple) else [got_h]
        worst = {"max_abs": 0.0, "max_rel": 0.0, "rel_to_range": 0.0}
        for i, (r, g) in enumerate(zip(ref_outputs if ref_outputs is not None else [reference], outs_h)):
            g = rope3_cmp(i, g)
            m = metrics(r, g)
            emit(f"torch_half_out{i}_vs_ref", fmt(m))
            for kk in worst:
                worst[kk] = max(worst[kk], m[kk])
        emit("torch_half_worst", fmt({**worst, "rmse": 0.0, "cos": 1.0}))

    if args.dry_run:
        print("[ov] DRY-RUN done (no ONNX).", flush=True)
        return

    # ---- ONNX -> IR -------------------------------------------------------------
    onnx_dir = out_root / "onnx"
    exp = export(wrapper, sample, onnx_dir, names, tag)
    emit("onnx", f"{exp['onnx']}  {exp['onnx_mb']:.1f} MB in {exp['n_files']} files " f"(graph + external data)  export {exp['export_s']:.1f}s")
    ir = to_ir(exp["onnx"], out_root, tag, fp16, compress=compress, fp16_io=bool(args.fp16_io))
    emit(
        "ir", f"{ir['xml']}  ops={ir['n_ops']}  bin={ir['bin_mb']:.1f} MB  " f"read={ir['read_s']:.1f}s save={ir['save_s']:.1f}s reload={ir['read_ir_s']:.1f}s"
    )
    emit("fp16_io", ir["fp16_io"] if ir["fp16_io"] is not None else "off (f32 interface)")
    if args.unit == "vae" and not args.vae_fold:
        emit("vae_fold", "DISABLED (--no-vae-fold): unfolded pre-promotion VAE graph")
    if args.unit == "vae" and args.vae_fold:
        # depth-tap fold, applied to the IR just written (same filenames) and re-verified
        ir, fmeta = apply_vae_fold(ir, compress=compress)
        emit(
            "vae_fold",
            f"{fmeta['macs_before_gmac']:.1f} -> {fmeta['macs_after_gmac']:.1f} GMAC "
            f"({fmeta['mac_reduction']:.2f}x), {fmeta['convs_folded']} convs folded, "
            f"ops {fmeta['ops_before']} -> {fmeta['ops_after']}, "
            f"bin {fmeta['bin_mb']:.1f} MB, kd3 left {len(fmeta['kd3_convs_remaining'])}",
        )
        emit("vae_fold_detail", fmeta)
    if args.unit == "action" and args.action_int8:
        # B1 split first (it parses the fp16 graph), then INT8 both graphs in place
        ir, imeta = apply_action_int8(ir, compress=compress, out_root=out_root)
        emit(
            "action_int8",
            "nncf {} | split {:.2f}s -> b1+ctx | action_fp16 {:.1f}->{:.1f} MB "
            "(x{:.3f}, {}/{} u8, {}/{} dequant-B) | action_b1 {:.1f}->{:.1f} MB "
            "(x{:.3f}, {}/{} u8, {}/{} dequant-B) "
            "| ctx fp16 {:.1f} MB | fp16 archive {}".format(
                imeta["nncf"],
                imeta["split_s"],
                imeta["action_fp16"]["bin_mb_before"],
                imeta["action_fp16"]["bin_mb_after"],
                imeta["action_fp16"]["ratio"],
                imeta["action_fp16"]["i8_weights"],
                imeta["action_fp16"]["matmuls"],
                imeta["action_fp16"]["dequant_b"],
                imeta["action_fp16"]["matmuls"],
                imeta["action_b1"]["bin_mb_before"],
                imeta["action_b1"]["bin_mb_after"],
                imeta["action_b1"]["ratio"],
                imeta["action_b1"]["i8_weights"],
                imeta["action_b1"]["matmuls"],
                imeta["action_b1"]["dequant_b"],
                imeta["action_b1"]["matmuls"],
                imeta["action_ctx_fp16_mb"],
                imeta["archive_dir"] or "none (fp16 dropped)",
            ),
        )
        emit("action_int8_detail", imeta)
        ir = dict(ir, model=ir["core"].read_model(str(ir["xml"])))  # further checks reload INT8
    elif args.unit == "action":
        emit("action_int8", "DISABLED (--no-action-int8): fp16 action graph, no action_b1 written")
    if args.unit == "video" and args.video_int8:
        # no split step for the video unit: export -> compress -> same filenames
        ir, vmeta = apply_video_int8(ir, compress=compress, out_root=out_root, archive_dir=args.video_int8_archive)
        v = vmeta["video_fp16"]
        emit(
            "video_int8",
            "nncf {} | video_fp16 {:.1f}->{:.1f} MB (x{:.3f}, {}/{} u8, {}/{} "
            "dequant-B, {} ignored kept {}, matmuls {}->{}) | fp16 archive {} | "
            "{:.1f}s total".format(
                vmeta["nncf"],
                v["bin_mb_before"],
                v["bin_mb_after"],
                v["ratio"],
                v["i8_weights"],
                v["matmuls"],
                v["dequant_b"],
                v["matmuls"],
                v["ignored_nodes"],
                v["kept_types"],
                v["matmuls_before"],
                v["matmuls_after"],
                vmeta["archive_dir"] or "none (fp16 dropped)",
                vmeta["total_s"],
            ),
        )
        emit("video_int8_detail", vmeta)
        ir = dict(ir, model=ir["core"].read_model(str(ir["xml"])))  # further checks reload INT8
    elif args.unit == "video":
        emit("video_int8", "DISABLED (--no-video-int8): fp16 video graph, 9381.8 MB bin")

    ordered = [t.contiguous().to(inp_dtype).numpy() for t in sample]  # order == names["in"]
    ordered = feed_arrays(ir["model"], [torch.from_numpy(a) for a in ordered])  # follow the DECLARED input element types

    if args.verify:
        refs = ref_outputs if ref_outputs is not None else [reference]
        for dev in [d.strip() for d in args.devices.split(",") if d.strip()]:
            try:
                cm = ir["core"].compile_model(ir["model"], dev)
                res = cm(ordered)
                outs = [torch.from_numpy(np.asarray(res[j]).copy()).float() for j in range(len(refs))]
                for j, (r, g) in enumerate(zip(refs, outs)):
                    m = metrics(r, rope3_cmp(j, g))
                    emit(f"numerics_{dev}_{prec}_out{j}", fmt(m))
                    log.setdefault(f"_max_{dev}", []).append(m)
                if dev == "CPU":
                    log["_cpu_out"] = outs[-1]
                del cm
            except Exception as exc:  # noqa: BLE001
                emit(f"numerics_{dev}_{prec}", f"FAILED {type(exc).__name__}: {str(exc)[:300]}")
        for dev in [d.strip() for d in args.devices.split(",") if d.strip()]:
            if f"_max_{dev}" in log:
                ms = log.pop(f"_max_{dev}")
                emit(
                    f"numerics_{dev}_WORST_OF_{len(ms)}",
                    "maxabs={:.3e} maxrel={:.3e} maxrel_over_outrange={:.3e} min_cos={:.9f}".format(
                        max(m["max_abs"] for m in ms), max(m["max_rel"] for m in ms), max(m["rel_to_range"] for m in ms), min(m["cos"] for m in ms)
                    ),
                )

    if args.fp64_probe and not args.trace_half:
        # Where does the residual error come from? Compare torch-fp64 against both the
        # torch-fp32 reference and the OpenVINO fp32 result.
        try:
            w64 = copy.deepcopy(wrapper).double()
            with torch.no_grad():
                got64 = w64(*[t.double() for t in sample])
            got64 = got64.float()
            emit("fp64_vs_fp32torch", fmt(metrics(reference, got64)))
            if "_cpu_out" in log:
                emit("fp64_vs_ovfp32", fmt(metrics(log["_cpu_out"], got64)))
            del w64
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as exc:  # noqa: BLE001
            emit("fp64_probe", f"FAILED {type(exc).__name__}: {str(exc)[:200]}")

    if args.bench:
        flop = None
        if args.unit == "action":
            n_act = sum(int(x.numel()) for x in model.action_expert.parameters())
            flop = 2.0 * n_act * ref["latents_action"].shape[1]  # 2 * params * tokens
            emit("action_expert_params", n_act)
        rows = {}
        for dev in [d.strip() for d in args.devices.split(",") if d.strip()]:
            r = bench_device(ir, dev, ordered, args.warmup, args.runs)
            if flop and "mean_ms" in r:
                r["tflops"] = round(flop / (r["mean_ms"] * 1e-3) / 1e12, 3)
            rows[dev] = r
            print(f"[ov] BENCH {args.unit} {args.dtype} {dev}: {json.dumps(r)}", flush=True)
        emit("bench", rows)
        if flop:
            emit("flop_per_step", flop)

    # the run log lands next to this script; the folder is created on first use
    (SCRIPTS / "logs").mkdir(exist_ok=True, parents=True)
    with open(SCRIPTS / "logs" / f"ov_{tag}{args.art_suffix}_{prec}.json", "w") as fh:
        json.dump(log, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
