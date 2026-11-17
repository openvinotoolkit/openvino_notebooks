# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Export the Wan2.2-TI2V-5B VAE *encoder* as a 4-D graph (library form, no CLI).

Why this is exact
-----------------
After the depth-tap fold every conv has kernel depth 1, and every tensor in the encoder has T == 1:
the only transient T == 2 tensors are the two `avg_shortcut` pads, which fold straight back into
channels.  The 5-D graph is therefore an image model carrying a dummy axis, and on the GPU it pays
for 5-D-only paths (`permute_ref`, `border_gpu_ref`, `reduce_ref`, blocked `b_fs_zyx_fsv16`
layouts).  This module rebuilds the same weights as a 2-D model: on an Arc B580 the VAE unit goes
5.51 -> 4.43 ms once the rank-4 RMS fusion is available in the plugin (see NOTES below).

What it does (torch level; the FasterWAM repo is not modified)
--------------------------------------------------------------
  CausalConv3d.forward   -> conv2d on the *live* depth tap `_padding[4]`.  With the causal front
                            pad and T == 1 the frame sits exactly there, so every other tap
                            multiplies exact zeros - the same invariant `fold_vae.py` exploits
                            at IR level.
  AvgDown3D.forward      -> avg_pool2d(factor_s) plus the causal zero frame(s) interleaved into the
                            channel axis (out_c = c*factor_t + ft, ft == 0 is the zero frame).
  RMS_norm.forward       -> broadcast shape rebuilt from the input rank (C at dim 1 / -1).
  AttentionBlock.forward -> (b,c,h,w).
  ResidualBlock / Encoder3d* / Resample
                         -> the 5-D feat_cache bookkeeping is dropped; a single frame never *reads*
                            a cache slot (each conv owns one and is visited once, which is also why
                            the shipping 5-D graph never sees a conv output with T > 1).

The wrapper keeps the 5-D *interface* (`[1,3,1,H,W] -> [1,48,1,H/16,W/16]`): the squeeze and the
unsqueeze are two Reshape nodes at the boundary, so `fwam_ov.py` and every harness keep feeding the
same tensors while everything between them is rank 4.

NOTES
-----
* Bit-for-bit equal maths at fp32 (max abs 4.4e-06, cos 1.000000000 against the 5-D encoder).
* The 4-D gain needs the rank-4 RMS fusion in the GPU plugin (a rank-4 `ReduceL2` otherwise lowers
  to `ocl:combined` and costs ~9x more per site, which more than eats the structural win).  With a
  stock plugin the 4-D graph is still a win on the iGPU (-15.6 %) and a small loss on the B580.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fasterwam.models.wan22 import wan_video_vae as wvv
from fasterwam.models.wan22.wan_video_vae import count_conv3d, patchify

_CACHE_T = 2
_CausalConv = None
_ResBlock = None


def _conv_forward_4d(self, x, cache_x=None):
    """CausalConv3d on a T == 1 input -> conv2d on the single live depth tap."""
    kd = int(self.kernel_size[0])
    tap = int(self._padding[4])  # causal front pad in T = index of the real frame
    if not 0 <= tap < kd:
        raise RuntimeError(f"VAE 4D: unexpected causal pad {tap} for kernel depth {kd}")
    pw, ph = int(self._padding[0]), int(self._padding[2])
    if pw or ph:
        x = F.pad(x, (pw, pw, ph, ph))
    return F.conv2d(
        x,
        self.weight[:, :, tap],
        self.bias,
        stride=tuple(int(v) for v in self.stride[1:]),
        padding=0,
        dilation=tuple(int(v) for v in self.dilation[1:]),
        groups=int(self.groups),
    )


def _rms_forward_4d(self, x):
    # gamma/bias are authored for the rank they were built with, so rebuild the broadcast shape from
    # the actual input rank: the channel axis stays at position 1 (or -1), it is NOT appended.
    shape = [1] * x.dim()
    shape[1 if self.channel_first else -1] = -1
    gamma = self.gamma.reshape(shape)
    bias = self.bias.reshape(shape) if torch.is_tensor(self.bias) else self.bias
    return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * gamma + bias


def _attention_forward_4d(self, x):
    identity = x
    b, c, h, w = x.size()
    y = self.norm(x)
    q, k, v = self.to_qkv(y).reshape(b, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
    y = F.scaled_dot_product_attention(q, k, v)
    y = y.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
    return self.proj(y) + identity


def _avgdown_forward_4d(self, x):
    if int(self.group_size) != int(self.factor_s) ** 2:
        raise RuntimeError(
            f"VAE 4D: avg_shortcut group_size {self.group_size} != factor_s^2 ({int(self.factor_s) ** 2}); "
            "the 4-D form assumes the mean is taken over the spatial window only"
        )
    y = F.avg_pool2d(x, int(self.factor_s), int(self.factor_s)) if self.factor_s > 1 else x
    if int(self.factor_t) == 1:
        return y
    # out_c = c*factor_t + ft, ft == 0 is the causal zero frame, the real frame is the last tap
    zeros = torch.zeros_like(y)
    return torch.stack([zeros] * (int(self.factor_t) - 1) + [y], dim=2).flatten(1, 2)


def _resample_forward_4d(self, x, feat_cache=None, feat_idx=(0,)):
    if self.mode == "upsample3d":
        raise RuntimeError("VAE 4D: upsample3d is decoder-only")
    y = self.resample(x)
    if self.mode == "downsample3d" and feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:  # first visit: the temporal time_conv is skipped
            feat_cache[idx] = y.clone()
            feat_idx[0] += 1
    return y, feat_cache, feat_idx


def _resblock_forward_4d(self, x, feat_cache=None, feat_idx=(0,)):
    h = self.shortcut(x)
    for layer in self.residual:
        x = layer(x, None) if isinstance(layer, _CausalConv) else layer(x)
    return x + h, feat_cache, feat_idx


def _encoder_forward_4d(self, x, feat_cache=None, feat_idx=(0,)):
    x = self.conv1(x)
    for layer in self.downsamples:
        out = layer(x, feat_cache, feat_idx)
        x = out[0] if isinstance(out, tuple) else out
    for layer in self.middle:
        out = layer(x, feat_cache, feat_idx) if isinstance(layer, _ResBlock) else layer(x)
        x = out[0] if isinstance(out, tuple) else out
    for layer in self.head:
        x = layer(x, None) if isinstance(layer, _CausalConv) else layer(x)
    return x, feat_cache, feat_idx


def patch_4d() -> None:
    """Monkeypatch the Wan VAE modules to operate on 4-D tensors (idempotent)."""
    global _CausalConv, _ResBlock
    _CausalConv, _ResBlock = wvv.CausalConv3d, wvv.ResidualBlock
    targets = [
        (wvv.CausalConv3d, _conv_forward_4d),
        (wvv.RMS_norm, _rms_forward_4d),
        (wvv.AttentionBlock, _attention_forward_4d),
        (wvv.AvgDown3D, _avgdown_forward_4d),
        (wvv.Resample, _resample_forward_4d),
        (wvv.ResidualBlock, _resblock_forward_4d),
    ]
    for name in ("Encoder3d", "Encoder3d_38"):
        enc = getattr(wvv, name, None)
        if enc is not None:
            targets.append((enc, _encoder_forward_4d))
    for cls, fn in targets:
        if getattr(cls.forward, "__name__", "") != fn.__name__:
            setattr(cls, "forward", fn)


class VaeEncode1_4D(nn.Module):
    """Single-frame Wan2.2 VAE encode with 4-D internals: `[1,3,1,H,W] -> [1,48,1,H/16,W/16]`.

    Same weights, same interface and same output as `export_openvino.VaeEncode1`; only the tensors
    inside the encoder are rank 4 (the boundary squeeze/unsqueeze are two Reshape nodes).
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self._n_conv = count_conv3d(vae.model.encoder)
        self.register_buffer("mean", vae.mean.clone().float(), persistent=False)
        self.register_buffer("inv_std", (1.0 / vae.std.clone().float()), persistent=False)

    def forward(self, x):  # [1,3,1,H,W]
        x = x.squeeze(2)  # -> [1,3,H,W]; the encoder runs rank 4 from here on
        xm = patchify(x, patch_size=2)  # [1,12,H/2,W/2]
        out = self.vae.model.encoder(xm, feat_cache=[None] * self._n_conv, feat_idx=[0])[0]
        mu, _logvar = self.vae.model.conv1(out).chunk(2, dim=1)
        z = self.mean.shape[0]
        mu = (mu - self.mean.view(1, z, 1, 1)) * self.inv_std.view(1, z, 1, 1)
        return mu.unsqueeze(2)  # back to [1,48,1,H/16,W/16]
