"""Self-contained V-JEPA 2.1 encoder definition, checkpoint loader and video preprocessing.

The model graph is built in a form that maps cleanly onto OpenVINO ops:
  * 3D-factored RoPE is baked into two constant full-width tables and applied
    as a single fused multiply-add (see `build_rope_tables`).
  * q/k/v use three separate Linear layers so the runtime can fuse
    Linear -> view -> transpose -> SDPA.
Both choices are numerically exact - see the notebook's accuracy section.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Input geometry (fixed by the released checkpoints) ----------------------
PATCH_SIZE = 16
TUBELET_SIZE = 2
NUM_FRAMES = 8
IMG_SIZE = 384

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
# Architecture configuration
# ============================================================================
def make_arch_config(embed_dim: int, depth: int, num_heads: int, mlp_hidden: int) -> dict:
    """Derive every shape the model needs from the four core hyperparameters."""
    head_dim = embed_dim // num_heads
    t = NUM_FRAMES // TUBELET_SIZE
    h_patches = IMG_SIZE // PATCH_SIZE
    w_patches = IMG_SIZE // PATCH_SIZE
    return dict(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_hidden=mlp_hidden,
        head_dim=head_dim,
        t=t,
        h_patches=h_patches,
        w_patches=w_patches,
        n_tokens=t * h_patches * w_patches,
        # RoPE positions were trained on a 256px grid; spatial positions are
        # rescaled onto it so a 384px input reuses the pretrained frequencies.
        pretrained_grid=256 // PATCH_SIZE,
        # Width of each of the three (depth/height/width) RoPE components.
        d_comp=2 * ((head_dim // 3) // 2),
    )


def detect_arch_from_state_dict(sd: dict, verbose: bool = True) -> dict:
    """Infer the architecture from checkpoint tensor shapes.

    Avoids hardcoding a single model size: the same notebook works for the
    300M (ViT-L) and 1B+ (ViT-g) encoders.
    """
    embed_dim = sd["patch_embed.proj.weight"].shape[0]
    depth = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    num_heads = sd["blocks.0.attn.qkv.weight"].shape[0] // (3 * 64)
    mlp_hidden = sd["blocks.0.mlp.fc1.weight"].shape[0]
    if verbose:
        n_params = sum(v.numel() for v in sd.values()) / 1e6
        print(f"Detected architecture: embed_dim={embed_dim}, depth={depth}, " f"num_heads={num_heads}, mlp_hidden={mlp_hidden}  (~{n_params:.0f}M params)")
    return make_arch_config(embed_dim, depth, num_heads, mlp_hidden)


# ============================================================================
# Rotary position embedding (3D factored)
# ============================================================================
def build_rope_tables(cfg: dict):
    """Precompute full-width ``[N, head_dim]`` RoPE cos/sin tables.

    The reference implementation slices q/k into three ``d_comp``-wide axis
    components (depth / height / width), rotates each with its own position
    vector, then concatenates them back together with an unrotated tail.

    Instead we bake ONE cos and ONE sin table of width ``head_dim``::

        lanes [0        : dd      ]  <- depth  (frame index) positions
        lanes [dd       : 2*dd    ]  <- height positions
        lanes [2*dd     : 3*dd    ]  <- width  positions
        lanes [3*dd     : head_dim]  <- cos = 1, sin = 0   (identity)

    Because the tail lanes hold cos=1 / sin=0, the fused expression
    ``x * cos + rotate_half(x) * sin`` reduces to exactly ``x`` there, so the
    fused form is an algebraic identity with the sliced form.

    ``rotate_half`` is additionally folded into a constant index permutation:
    it maps ``y[2i] = -x[2i+1]``, ``y[2i+1] = x[2i]`` - a pairwise swap plus a
    -1/+1 sign pattern. Folding the sign into the sin table turns the whole
    thing into one Gather::

        out = x * cos + gather(x, perm) * sin_signed

    Negating a constant rather than an activation is lossless in IEEE-754, so
    this stays bit-exact.

    Returns:
        (cos, sin_signed, perm) - float32 ``[N, head_dim]`` tables and an
        int64 ``[head_dim]`` permutation.
    """
    n, head_dim, dd = cfg["n_tokens"], cfg["head_dim"], cfg["d_comp"]
    hp, wp, pg = cfg["h_patches"], cfg["w_patches"], cfg["pretrained_grid"]

    # Recover (frame, row, col) for every flattened token index.
    ids = torch.arange(n, dtype=torch.float32)
    tokens_per_frame = hp * wp
    frame = torch.div(ids, tokens_per_frame, rounding_mode="floor")
    rem = ids - tokens_per_frame * frame
    height = torch.div(rem, wp, rounding_mode="floor")
    width = rem - wp * height
    positions = [frame, height * (pg - 1) / (hp - 1), width * (pg - 1) / (wp - 1)]

    # Identity defaults so untouched tail lanes are a no-op.
    cos = torch.ones(n, head_dim, dtype=torch.float32)
    sin = torch.zeros(n, head_dim, dtype=torch.float32)

    omega = 1.0 / (10000 ** (torch.arange(dd // 2, dtype=torch.float32) / (dd / 2.0)))
    for axis, pos in enumerate(positions):
        freq = torch.einsum("n,f->nf", pos, omega)
        cos[:, axis * dd : (axis + 1) * dd] = freq.cos().repeat_interleave(2, dim=-1)
        sin[:, axis * dd : (axis + 1) * dd] = freq.sin().repeat_interleave(2, dim=-1)

    perm = torch.arange(head_dim, dtype=torch.int64).view(-1, 2).flip(-1).flatten()
    signs = torch.ones(head_dim, dtype=torch.float32)
    signs[0::2] = -1.0
    return cos, sin * signs, perm


# ============================================================================
# Model
# ============================================================================
class RoPEAttention(nn.Module):
    """Multi-head attention with 3D factored RoPE and native SDPA."""

    def __init__(self, cfg: dict):
        super().__init__()
        dim, num_heads = cfg["embed_dim"], cfg["num_heads"]
        self.embed_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_tokens = cfg["n_tokens"]

        # Separate projections, populated from the checkpoint's fused qkv.
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

        cos, sin, perm = build_rope_tables(cfg)
        # [1, 1, N, head_dim] broadcasts over [B, num_heads, N, head_dim].
        # persistent=False keeps these out of state_dict; they become IR
        # Constants at conversion time.
        self.register_buffer("rope_cos", cos[None, None], persistent=False)
        self.register_buffer("rope_sin", sin[None, None], persistent=False)
        self.register_buffer("rope_perm", perm, persistent=False)

    def _apply_rope(self, t):
        swapped = torch.index_select(t, -1, self.rope_perm)
        return t * self.rope_cos + swapped * self.rope_sin

    def forward(self, x):
        # Static shapes throughout - avoids ShapeOf/Range ops in the IR.
        n, nh, hd, c = self.n_tokens, self.num_heads, self.head_dim, self.embed_dim
        q = self.q_proj(x).view(1, n, nh, hd).transpose(1, 2)
        k = self.k_proj(x).view(1, n, nh, hd).transpose(1, 2)
        v = self.v_proj(x).view(1, n, nh, hd).transpose(1, 2)
        q, k = self._apply_rope(q), self._apply_rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(1, n, c))


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.fc1 = nn.Linear(cfg["embed_dim"], cfg["mlp_hidden"])
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg["mlp_hidden"], cfg["embed_dim"])

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim = cfg["embed_dim"]
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = RoPEAttention(cfg)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed3D(nn.Module):
    """Tubelet tokenizer: a strided 3D convolution over (time, height, width)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels=3,
            out_channels=cfg["embed_dim"],
            kernel_size=(TUBELET_SIZE, PATCH_SIZE, PATCH_SIZE),
            stride=(TUBELET_SIZE, PATCH_SIZE, PATCH_SIZE),
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VJEPAEncoder(nn.Module):
    """Container whose submodule names mirror the checkpoint keys."""

    def __init__(self, cfg: dict):
        super().__init__()
        dim = cfg["embed_dim"]
        self.patch_embed = PatchEmbed3D(cfg)
        self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["depth"])])
        self.norms_block = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(4)])


class VJEPAEncoderForExport(nn.Module):
    """Traceable end-to-end graph: raw video clip -> encoder tokens."""

    def __init__(self, encoder: VJEPAEncoder):
        super().__init__()
        self.patch_embed = encoder.patch_embed
        self.video_mod_embed = encoder.video_mod_embed
        self.blocks = encoder.blocks
        self.final_norm = encoder.norms_block[-1]

    def forward(self, x):
        x = self.patch_embed(x) + self.video_mod_embed
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)


# ============================================================================
# Checkpoint loading
# ============================================================================
def split_qkv_weights(state_dict: dict, cfg: dict) -> dict:
    """Rewrite each fused ``attn.qkv.{weight,bias}`` into separate q/k/v tensors.

    The checkpoint stores one ``[3*dim, dim]`` matrix whose row blocks are
    exactly q, k, v in order - the reference forward simply reshapes it apart.
    Slicing the rows is therefore an exact, lossless re-packing.
    """
    dim = cfg["embed_dim"]
    out = {}
    for k, v in state_dict.items():
        if k.endswith("attn.qkv.weight") or k.endswith("attn.qkv.bias"):
            base, suffix = k.rsplit("qkv.", 1)
            out[f"{base}q_proj.{suffix}"] = v[0:dim]
            out[f"{base}k_proj.{suffix}"] = v[dim : 2 * dim]
            out[f"{base}v_proj.{suffix}"] = v[2 * dim : 3 * dim]
        else:
            out[k] = v
    return out


def load_encoder(checkpoint_path, verbose: bool = True):
    """Build the encoder and load weights. Returns ``(model, cfg)``."""
    sd = torch.load(str(checkpoint_path), weights_only=True, map_location="cpu")
    key = "target_encoder" if "target_encoder" in sd else "encoder"
    enc_sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd[key].items()}

    cfg = detect_arch_from_state_dict(enc_sd, verbose=verbose)
    encoder = VJEPAEncoder(cfg)
    encoder.load_state_dict(split_qkv_weights(enc_sd, cfg), strict=False)
    encoder.eval().float()
    return VJEPAEncoderForExport(encoder).eval(), cfg


def find_checkpoint(search_dirs=(".", "pretrained", "../pretrained")):
    """Locate a V-JEPA 2.1 ``.pt`` checkpoint in the usual places.

    Prefers the smallest checkpoint found, so a tutorial run picks the lighter
    encoder when several sizes are available.
    """
    found = []
    for d in search_dirs:
        if Path(d).is_dir():
            found.extend(Path(d).glob("vjepa2*.pt"))
    return min(found, key=lambda p: p.stat().st_size) if found else None


# ============================================================================
# Video preprocessing
# ============================================================================
def load_video_clip(video_path, num_frames: int = NUM_FRAMES, img_size: int = IMG_SIZE):
    """Sample a clip and preprocess it into the encoder's input tensor.

    Steps: sample ``num_frames`` frames uniformly across the whole video ->
    resize the short side (aspect-preserving, bilinear) -> center crop ->
    scale to [0, 1] -> normalize with ImageNet statistics.

    Returns:
        (clip, frames) where ``clip`` is a float32 ``[1, 3, T, H, W]`` array
        ready for the model and ``frames`` is a uint8 ``[T, H, W, 3]`` array of
        the cropped RGB frames for display.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    wanted = np.linspace(0, total - 1, num_frames).astype(int)

    frames, idx, next_i = [], 0, 0
    while next_i < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        while next_i < len(wanted) and wanted[next_i] == idx:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            next_i += 1
        idx += 1
    cap.release()
    while len(frames) < num_frames:  # short/truncated videos
        frames.append(frames[-1])

    # Resize short side, preserving aspect ratio.
    short_side = int(256.0 / 224.0 * img_size)
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        if w < h:
            new_w, new_h = short_side, int(round(short_side * h / w))
        else:
            new_h, new_w = short_side, int(round(short_side * w / h))
        resized.append(cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR))

    # Center crop.
    cropped = []
    for f in resized:
        h, w = f.shape[:2]
        y1, x1 = int(round((h - img_size) / 2.0)), int(round((w - img_size) / 2.0))
        cropped.append(f[y1 : y1 + img_size, x1 : x1 + img_size])
    display_frames = np.stack(cropped).astype(np.uint8)

    # [T, H, W, C] -> [1, C, T, H, W], normalized.
    clip = display_frames.astype(np.float32) / 255.0
    clip = (clip - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    clip = clip.transpose(3, 0, 1, 2)[None]
    return np.ascontiguousarray(clip, dtype=np.float32), display_frames


# ============================================================================
# Metrics
# ============================================================================
def embedding_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Deviation of ``candidate`` embeddings from a ``reference`` tensor."""
    r = reference.astype(np.float64).reshape(-1)
    c = candidate.astype(np.float64).reshape(-1)

    rt = reference.astype(np.float64).reshape(-1, reference.shape[-1])
    ct = candidate.astype(np.float64).reshape(-1, candidate.shape[-1])
    per_token = np.sum(rt * ct, axis=1) / (np.linalg.norm(rt, axis=1) * np.linalg.norm(ct, axis=1) + 1e-30)

    err_power = float(np.sum((r - c) ** 2))
    return {
        "max_abs": float(np.abs(r - c).max()),
        "mean_abs": float(np.abs(r - c).mean()),
        "cosine": float(r @ c / (np.linalg.norm(r) * np.linalg.norm(c))),
        "worst_token_cosine": float(per_token.min()),
        "snr_db": 10.0 * np.log10(float(np.sum(r**2)) / err_power) if err_power > 0 else float("inf"),
    }
