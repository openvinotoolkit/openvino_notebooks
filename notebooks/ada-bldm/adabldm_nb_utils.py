"""Helpers for the AdaBLDM (ControlNet + Stable-Diffusion UNet) OpenVINO notebook.

AdaBLDM (https://github.com/GrandpaXun242/AdaBLDM) is a blended latent-diffusion model for
industrial defect generation: a ControlNet branch conditions a Stable-Diffusion v1.5 UNet on a
"triple map" (object mask + defect mask). Only the two networks are neural networks, so only they
are converted and quantized; the DDIM sampler and the triple-map construction stay on the host.

The upstream repository ships **no trained checkpoint** (``test.py`` points at a local
``lightning_logs/...`` training output). This notebook therefore follows the project's documented
*pretrain stage*: it builds the AdaBLDM architecture and initializes it from the public Stable
Diffusion v1.5 weights (the same initialization ``tool_add_control.py`` performs). The compute
graph - and therefore the latency, throughput and model size OpenVINO optimizes - is identical to
a fine-tuned AdaBLDM; only the weight values differ.
"""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent
REPO_URL = "https://github.com/GrandpaXun242/AdaBLDM.git"
REPO = ROOT / "AdaBLDM"
MODEL_YAML = "models/cldm_v15.yaml"
BASE_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_FILE = "v1-5-pruned-emaonly.ckpt"
CLIP_ID = "openai/clip-vit-large-patch14"

LATENT_HW = (32, 32)   # 256 px image / 8 (VAE downsample)
IMAGE_HW = (256, 256)
CONTEXT_LEN, CONTEXT_DIM = 77, 768


def find_file(name, env=None, subdirs=("", "weights", "models")):
    """First existing <base>/<subdir>/<name> for base in (notebook/, notebooks/, repo parent, ~/models)."""
    if env and os.environ.get(env):
        return Path(os.environ[env]).expanduser().resolve()
    for base in (NB_DIR, ROOT, ROOT.parent, Path.home() / "models"):
        for sd in subdirs:
            p = base / sd / name
            if p.exists():
                return p.resolve()
    return (ROOT / "weights" / name)


BASE_CKPT = find_file(BASE_FILE, "ADABLDM_BASE_CKPT")


def ensure_repo(repo=REPO):
    """Shallow-clone the unmodified AdaBLDM repository (cldm/ldm packages: model + sampler)."""
    if not (repo / "cldm" / "cldm.py").exists():
        import subprocess

        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {REPO_URL} -> {repo}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def setup_compat():
    """Compatibility shims so the 2023 codebase imports on modern numpy / pytorch_lightning."""
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if importlib.util.find_spec("pytorch_lightning.utilities.distributed") is None:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only

        _m = types.ModuleType("pytorch_lightning.utilities.distributed")
        _m.rank_zero_only = rank_zero_only
        sys.modules["pytorch_lightning.utilities.distributed"] = _m


def download_base() -> Path:
    """Fetch the public Stable-Diffusion v1.5 checkpoint used for the documented pretrain init."""
    if not BASE_CKPT.exists():
        from huggingface_hub import hf_hub_download

        BASE_CKPT.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {BASE_FILE} from huggingface.co/{BASE_REPO}")
        p = hf_hub_download(repo_id=BASE_REPO, filename=BASE_FILE, local_dir=str(BASE_CKPT.parent))
        return Path(p)
    return BASE_CKPT


def _get_node_name(name, parent):
    return (True, name[len(parent):]) if name.startswith(parent) else (False, "")


def build_denoiser(base_ckpt=BASE_CKPT, verbose=True):
    """Build the AdaBLDM denoiser (ControlNet + ControlledUnetModel) and apply the SD v1.5 pretrain init.

    Mirrors ``tool_add_control.py``: the UNet and the ControlNet's shared trunk (``input_blocks``,
    ``middle_block``, ``time_embed``) are copied from ``model.diffusion_model.*``; the ControlNet's
    zero convolutions, ``middle_block_out`` and ``input_hint_block`` are newly initialized (zero),
    exactly as in the upstream pretrain stage.
    """
    from omegaconf import OmegaConf

    from cldm.cldm import ControlNet, ControlledUnetModel

    cfg = OmegaConf.load(str(REPO / MODEL_YAML))
    p = cfg.model.params
    unet = ControlledUnetModel(**{k: v for k, v in p.unet_config.params.items() if k != "use_checkpoint"}).eval().float()
    cnet = ControlNet(**{k: v for k, v in p.control_stage_config.params.items() if k != "use_checkpoint"}).eval().float()

    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    unet_sd = {}
    for k, v in unet.state_dict().items():
        src = "model.diffusion_model." + k
        unet_sd[k] = sd[src].clone() if src in sd else v.clone()
    unet.load_state_dict(unet_sd, strict=True)

    cnet_sd, copied = {}, 0
    for k, v in cnet.state_dict().items():
        src = "model.diffusion_model." + k if k.startswith(("input_blocks.", "middle_block.", "time_embed.")) else None
        if src and src in sd:
            cnet_sd[k] = sd[src].clone(); copied += 1
        else:
            cnet_sd[k] = v.clone()
    cnet.load_state_dict(cnet_sd, strict=True)
    if verbose:
        print(f"SD v1.5 init: UNet {sum(x.numel() for x in unet.parameters()) / 1e6:.1f}M, "
              f"ControlNet {sum(x.numel() for x in cnet.parameters()) / 1e6:.1f}M "
              f"({copied} ControlNet tensors copied from SD v1.5, zero convs newly initialized)")

    class Denoiser(torch.nn.Module):
        """ControlNet + ControlledUnetModel fused into one graph: (latent, hint, t, context) -> noise."""

        def __init__(self, cnet, unet):
            super().__init__()
            self.cnet = cnet
            self.unet = unet

        def forward(self, x, hint, t, context):
            control = list(self.cnet(x, hint, t, context))
            return self.unet(x, t, context, control=control)

    Denoiser.__module__ = __name__
    Denoiser.__qualname__ = "Denoiser"
    return Denoiser(cnet, unet).eval().float()


def encode_prompt(text, device="cpu"):
    """CLIP text embedding [1, 77, 768] for a prompt (the model's cross-attention conditioning)."""
    from transformers import CLIPTokenizer, CLIPTextModel

    tok = CLIPTokenizer.from_pretrained(CLIP_ID)
    enc = CLIPTextModel.from_pretrained(CLIP_ID).eval()
    tokens = tok(text, padding="max_length", max_length=CONTEXT_LEN, truncation=True, return_tensors="pt")
    with torch.no_grad():
        return enc(**tokens).last_hidden_state.to(device).float()


def build_vae(base_ckpt=BASE_CKPT):
    """SD v1.5 first-stage VAE (encoder/decoder) for turning latents into pixels."""
    from omegaconf import OmegaConf

    from ldm.models.autoencoder import AutoencoderKL

    cfg = OmegaConf.load(str(REPO / MODEL_YAML))
    fc = cfg.model.params.first_stage_config.params
    vae = AutoencoderKL(ddconfig=fc.ddconfig, embed_dim=4, lossconfig=fc.lossconfig).eval().float()
    sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    vsd = {k[len("first_stage_model."):]: v.clone() for k, v in sd.items()
           if k.startswith("first_stage_model.") and "loss" not in k}
    vae.load_state_dict(vsd, strict=False)
    return vae


def make_inputs(context, seed=0):
    """A denoiser input tuple: (latent, hint/triple-map, timestep, context)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 4, *LATENT_HW, generator=g)
    hint = torch.zeros(1, 3, *IMAGE_HW)
    t = torch.full((1,), 500.0)
    return x, hint, t, context


def static_shapes(model):
    """Reshape the traced denoiser to fully static shapes (benchmark_app requirement)."""
    shapes = [(1, 4, *LATENT_HW), (1, 3, *IMAGE_HW), (1,), (1, CONTEXT_LEN, CONTEXT_DIM)]
    model.reshape({i.get_any_name(): s for i, s in zip(model.inputs, shapes)})
    return model


def torch_runner(model):
    def run(*inputs):
        with torch.no_grad():
            return model(*[torch.as_tensor(i) for i in inputs]).numpy()
    return run


def ov_runner(core, xml_or_model, device="CPU", **props):
    cm = core.compile_model(xml_or_model, device, {"PERFORMANCE_HINT": "LATENCY", **props})
    out = cm.output(0)

    def run(*inputs):
        return cm([np.ascontiguousarray(i) for i in inputs])[out]
    run.exec_precision = cm.get_property("INFERENCE_PRECISION_HINT")
    return run


def timeit(run, *inputs, n=10, warmup=3):
    import time

    for _ in range(warmup):
        run(*inputs)
    t0 = time.perf_counter()
    for _ in range(n):
        y = run(*inputs)
    return y, (time.perf_counter() - t0) * 1000 / n


def cosine(a, b):
    a = np.asarray(a).reshape(-1).astype(np.float64)
    b = np.asarray(b).reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def parse_benchmark_log(path):
    """Extract average latency, FPS and nireq from a benchmark_app log."""
    text = Path(path).read_text()
    avg = fps = nireq = None
    for line in text.splitlines():
        if "Average:" in line:
            avg = float(line.split(":")[-1].strip().split()[0])
        elif "Throughput:" in line:
            fps = float(line.split(":")[-1].strip().split()[0])
        elif "nireq" in line and "=" in line:
            try:
                nireq = int(line.split("=")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return {"avg_ms": round(avg, 2) if avg else None, "fps": round(fps, 2) if fps else None, "nireq": nireq}
