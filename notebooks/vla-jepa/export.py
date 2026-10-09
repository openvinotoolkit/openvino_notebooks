"""
OpenVINO export for VLA-JEPA.

Exports the two inference-time subgraphs to ``openvino_model/``:

  1. Qwen3-VL-2B  -> openvino_model/qwen3vl/   (optimum-intel multi-IR export)
  2. DiT action head (single flow-matching step) -> openvino_model/action_dit.xml
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import openvino as ov
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

DIT_PRESETS = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """(B, T) timesteps -> (B, T, embedding_dim). Note: sin first, then cos."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (torch.log(torch.tensor(10000.0)) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class MLP(nn.Module):
    """layer2(relu(layer1(x))). Used for both state_encoder and action_decoder."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # actions (B, T, action_dim); timesteps (B,) -> replicate across T
        t = timesteps.unsqueeze(1).expand(-1, actions.shape[1])
        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(t).to(dtype=a_emb.dtype)
        x = swish(self.layer2(torch.cat([a_emb, tau_emb], dim=-1)))
        return self.layer3(x)


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class AdaLayerNorm(nn.Module):
    """Note the chunk order is (scale, shift), not the more common (shift, scale)."""

    def __init__(self, embedding_dim: int, norm_eps: float = 1e-5, norm_elementwise_affine: bool = False):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """Single-attention block: AdaLayerNorm -> attn1 -> LayerNorm -> FeedForward.

    ``attn1`` is cross-attention when ``cross_attention_dim`` is set, otherwise
    self-attention. There is no ``attn2``.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
    ):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=True,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout, bias=True)

    def forward(self, hidden_states, encoder_hidden_states=None, temb=None):
        norm_hidden_states = self.norm1(hidden_states, temb)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = attn_output + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class DiT(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        interleave_self_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.interleave_self_attention = interleave_self_attention
        self.inner_dim = num_attention_heads * attention_head_dim

        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim)

        blocks = []
        for idx in range(num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            blocks.append(
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    final_dropout=final_dropout,
                    cross_attention_dim=None if use_self_attn else cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        temb = self.timestep_encoder(timestep)
        for idx, block in enumerate(self.transformer_blocks):
            use_self_attn = idx % 2 == 1 and self.interleave_self_attention
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=None if use_self_attn else encoder_hidden_states,
                temb=temb,
            )
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


class FlowmatchingActionHeadSingleStep(nn.Module):
    """One flow-matching denoising step, exported as a single OpenVINO model.

    (noisy_actions, timestep, embodied_tokens, state) -> velocity [B, horizon, action_dim]
    """

    def __init__(self, cfg: dict):
        super().__init__()
        preset = DIT_PRESETS[cfg["action_model_type"]]
        dit_cfg = {**preset, **cfg["diffusion_model_cfg"]}
        dit_cfg.pop("input_embedding_dim", None)

        self.input_embedding_dim = preset["input_embedding_dim"]
        self.hidden_size = cfg["hidden_size"]
        self.action_horizon = cfg["action_horizon"]
        self.add_pos_embed = cfg["add_pos_embed"]

        self.model = DiT(**dit_cfg)
        self.state_encoder = MLP(cfg["state_dim"], self.hidden_size, self.input_embedding_dim)
        self.action_encoder = ActionEncoder(cfg["action_dim"], self.input_embedding_dim)
        self.action_decoder = MLP(self.hidden_size, self.hidden_size, cfg["action_dim"])
        self.future_tokens = nn.Embedding(cfg["num_target_vision_tokens"], self.input_embedding_dim)
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(cfg["max_seq_len"], self.input_embedding_dim)

    def forward(self, noisy_actions, timestep, embodied_tokens, state):
        state_features = self.state_encoder(state)  # [B, 1, 768]
        action_features = self.action_encoder(noisy_actions, timestep)  # [B, 7, 768]

        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(embodied_tokens.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=embodied_tokens,
            timestep=timestep,
        )
        pred = self.action_decoder(model_output)
        return pred[:, -self.action_horizon :]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def audit_ops(ov_model: ov.Model, name: str) -> None:
    """Fail loudly on unresolved PyTorch ops (the only case needing a custom kernel)."""
    fw_nodes = [op for op in ov_model.get_ordered_ops() if op.get_type_name() == "FrameworkNode"]
    if fw_nodes:
        raise RuntimeError(f"{name}: {len(fw_nodes)} unresolved FrameworkNode(s): " f"{[n.get_friendly_name() for n in fw_nodes][:10]}")
    print(f"  [{name}] op audit clean — {len(set(op.get_type_name() for op in ov_model.get_ordered_ops()))} distinct op types, no FrameworkNode")


def load_action_cfg(config_yaml: Path) -> dict:
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(config_yaml), resolve=True)

    def find(node):
        if isinstance(node, dict):
            if "action_model_type" in node and "diffusion_model_cfg" in node:
                return node
            for v in node.values():
                found = find(v)
                if found is not None:
                    return found
        return None

    cfg = find(raw)
    if cfg is None:
        raise RuntimeError(f"could not locate the action_model block in {config_yaml}")
    return cfg


# --------------------------------------------------------------------------- #
# Part B — DiT action head
# --------------------------------------------------------------------------- #


def export_dit(args) -> None:
    print("\n=== Exporting DiT action head ===")
    cfg = load_action_cfg(Path(args.config))

    head = FlowmatchingActionHeadSingleStep(cfg).eval()

    print("  loading fine-tuned weights from checkpoint …")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd = {k[len("action_model.") :]: v.float() for k, v in ckpt.items() if k.startswith("action_model.")}
    missing, unexpected = head.load_state_dict(sd, strict=False)
    # position_embedding/future_tokens are always present; anything else missing is a bug.
    if missing:
        raise RuntimeError(f"missing keys when loading DiT weights: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading DiT weights: {unexpected[:10]}")
    print(f"  loaded {len(sd)} tensors (0 missing, 0 unexpected)")
    del ckpt
    gc.collect()

    B = 1
    horizon, action_dim = cfg["action_horizon"], cfg["action_dim"]
    n_tok = cfg["num_target_vision_tokens"]
    example = (
        torch.randn(B, horizon, action_dim),
        torch.tensor([250], dtype=torch.int64),
        torch.randn(B, n_tok, 2048),
        torch.randn(B, 1, cfg["state_dim"]),
    )

    print("  tracing …")
    with torch.no_grad():
        ov_model = ov.convert_model(head, example_input=example)

    names = ["noisy_actions", "timestep", "embodied_tokens", "state"]
    for port, name in zip(ov_model.inputs, names):
        port.get_tensor().set_names({name})
    ov_model.outputs[0].get_tensor().set_names({"velocity"})

    audit_ops(ov_model, "action_dit")

    out_xml = Path(args.output_dir) / "action_dit.xml"
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    # Kept at fp32. The head is only 150M params (~600 MB), so fp16 buys little,
    # and its last output column is a gripper logit thresholded at 0.5 that the
    # golden trajectory approaches to within 0.04 — fp16 rounding alone is enough
    # to flip that bit on one device and not another.
    ov.save_model(ov_model, out_xml, compress_to_fp16=False)
    print(f"  saved {out_xml} ({out_xml.with_suffix('.bin').stat().st_size / 1e6:.1f} MB)")

    validate_dit(args, head, out_xml, cfg)


def validate_dit(args, head, out_xml: Path, cfg: dict) -> None:
    golden = Path(args.golden_dir)
    if not (golden / "embodied_action_tokens.npy").exists():
        print("  ! golden outputs not found — skipping DiT validation")
        return

    print("\n  --- DiT validation vs golden (per-step velocity) ---")
    emb = np.load(golden / "embodied_action_tokens.npy").astype(np.float32)
    state = np.load(golden / "input_state.npy").astype(np.float32)
    if state.ndim == 2:
        state = state[:, None, :]  # [B, 1, state_dim]

    core = ov.Core()
    compiled = core.compile_model(str(out_xml), args.device)

    n_steps = cfg["num_inference_timesteps"]
    buckets = cfg["num_timestep_buckets"]
    all_pass = True
    for i in range(n_steps):
        noisy = np.load(golden / f"dit_noisy_actions_step{i}.npy").astype(np.float32)
        ref = np.load(golden / f"dit_velocity_step{i}.npy").astype(np.float32)
        t_disc = int(i / float(n_steps) * buckets)

        ov_out = compiled(
            {
                "noisy_actions": noisy,
                "timestep": np.array([t_disc], dtype=np.int64),
                "embodied_tokens": emb,
                "state": state,
            }
        )["velocity"]

        with torch.no_grad():
            pt_out = head(
                torch.from_numpy(noisy),
                torch.tensor([t_disc], dtype=torch.int64),
                torch.from_numpy(emb),
                torch.from_numpy(state),
            ).numpy()

        c_ov = cos_sim(ref, ov_out)
        c_pt = cos_sim(ref, pt_out)
        ok = c_ov >= 0.999
        all_pass &= ok
        print(f"    t={t_disc:4d}  cos(golden, OV)={c_ov:.6f}  " f"cos(golden, re-impl PyTorch)={c_pt:.6f}  {'PASS' if ok else 'FAIL'}")

    print(f"  DiT validation: {'ALL PASS' if all_pass else 'FAILURES PRESENT'}")


# --------------------------------------------------------------------------- #
# Part A — Qwen3-VL
# --------------------------------------------------------------------------- #


def export_qwen(args) -> None:
    print("\n=== Exporting Qwen3-VL-2B ===")
    from transformers import AutoConfig, AutoProcessor, Qwen3VLModel
    from optimum.exporters.openvino import export_from_model

    out_dir = Path(args.output_dir) / "qwen3vl"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.qwen_base)
    cfg.torch_dtype = "float32"

    print("  building Qwen3VLModel (bare — no lm_head, so the language IR emits hidden states) …")
    model = Qwen3VLModel(cfg)

    print("  loading fine-tuned weights from checkpoint …")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    prefix = "qwen_vl_interface.model.model."
    sd = {k[len(prefix) :]: v.float() for k, v in ckpt.items() if k.startswith(prefix)}
    if not sd:
        raise RuntimeError(f"no keys with prefix {prefix!r} found in checkpoint")

    emb_rows = sd["language_model.embed_tokens.weight"].shape[0]
    if emb_rows != model.get_input_embeddings().weight.shape[0]:
        print(f"  resizing token embeddings -> {emb_rows}")
        model.resize_token_embeddings(emb_rows)

    # This must stay strict. `optimum-cli export openvino -m <hf_dir>` silently
    # exports a RANDOMLY INITIALISED model here: the on-disk checkpoint was saved
    # from Qwen3VLForConditionalGeneration so its keys carry a `model.` prefix that
    # AutoModel/Qwen3VLModel does not expect, and transformers only *warns*
    # ("Some weights ... newly initialized") before continuing to a clean exit 0.
    # The resulting IR has perfect shapes and garbage numerics.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"missing keys loading Qwen weights: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys loading Qwen weights: {unexpected[:10]}")
    print(f"  loaded {len(sd)} tensors (0 missing, 0 unexpected)")
    del ckpt, sd
    gc.collect()

    # VLA-JEPA consumes `hidden_states[-1]` of Qwen3VLForConditionalGeneration, which is
    # the output of the LAST DECODER LAYER — i.e. the tensor *before* the text model's
    # final RMSNorm. Qwen3VLModel.last_hidden_state (what this IR emits) is the tensor
    # *after* it. Those are not interchangeable: per-token RMS here ranges from ~2.9 to
    # ~886 (massive-activation outliers), so the norm rescales tokens by wildly different
    # factors and the extracted embodied tokens land at cos=0.767 vs the golden.
    # RMSNorm discards the per-token scale, so this cannot be undone downstream —
    # the norm has to be removed from the exported graph.
    lm_norm = model.language_model.norm
    if type(lm_norm).__name__ != "Qwen3VLTextRMSNorm":
        raise RuntimeError(f"expected language_model.norm to be Qwen3VLTextRMSNorm, got {type(lm_norm).__name__}")
    model.language_model.norm = torch.nn.Identity()
    print("  replaced language_model.norm with Identity (export pre-norm hidden states)")

    model = model.eval().float()

    # stateful=True is the exporter default and the right choice here: VLA-JEPA runs
    # a single prefill pass and never decodes autoregressively, so the KV cache is
    # write-only. stateful=False materialises 56 `present.*` outputs that are copied
    # out of the graph on every call and immediately discarded.
    #
    # `export_from_model`'s `ov_config` only controls things like fp16 packing
    # (`OVConfig(dtype="fp16")`); its `quantization_config` is never read by the
    # export path itself — `_save_model()` ignores it entirely, so weights always
    # come out at full precision here regardless of what's passed. NNCF weight
    # compression only happens later, as a *load-time* step inside
    # `OVBaseModel._load_model` / `from_pretrained(quantization_config=...)`. So we
    # always export FP32 first, then (if requested) re-load + compress + re-save.
    print("  exporting (task=feature-extraction, stateful=True) — this takes a few minutes …")
    export_from_model(
        model,
        output=out_dir,
        task="feature-extraction",
        stateful=True,
    )
    del model
    gc.collect()

    if args.int4:
        from optimum.intel import OVModelForVisualCausalLM, OVWeightQuantizationConfig

        print("  compressing Qwen3-VL weights to INT4 …")
        quantization_config = OVWeightQuantizationConfig(bits=4, group_size=128, ratio=0.8)
        compressed = OVModelForVisualCausalLM.from_pretrained(out_dir, quantization_config=quantization_config, compile=False)
        # `from_pretrained` memory-maps the FP32 .bin files it just read (OpenVINO's
        # default `read_model` behaviour). Calling `save_pretrained(out_dir)` — i.e.
        # writing the compressed weights back into the SAME files that are still
        # mmap'd open — truncates/rewrites pages out from under that live mapping and
        # crashes the process with SIGBUS (exit code 135) partway through the write,
        # leaving a 0-byte .bin. Save to a scratch dir instead, then swap it in only
        # after the source model object (and its mmaps) has been released.
        tmp_dir = out_dir.parent / (out_dir.name + "_int4_tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        compressed.save_pretrained(tmp_dir)
        del compressed
        gc.collect()
        for f in tmp_dir.iterdir():
            shutil.move(str(f), str(out_dir / f.name))
        tmp_dir.rmdir()
        print("  INT4 weight compression complete")

    # Save the processor/tokenizer alongside so Step 3 can load everything from one dir.
    # The BASE tokenizer does not know VLA-JEPA's added tokens: without this the
    # string "<|embodied_action|>" tokenises into 13 ordinary subword tokens, no
    # position ever matches `embodied_action_token_id`, and Step 3 extracts nothing.
    # Tokens must be added in this exact order to reproduce the training ids
    # (action_0..action_27 -> 151669..151696, embodied_action -> 151697).
    try:
        proc = AutoProcessor.from_pretrained(args.qwen_base)
        tok = proc.tokenizer
        n_action_tokens = 28  # action_horizon * 4
        new_tokens = [f"<|action_{i}|>" for i in range(n_action_tokens)] + ["<|embodied_action|>"]
        added = tok.add_tokens([t for t in new_tokens if t not in tok.get_vocab()], special_tokens=True)
        emb_id = tok.convert_tokens_to_ids("<|embodied_action|>")
        print(f"  tokenizer: added {added} special tokens, len={len(tok)}, " f"<|embodied_action|> -> {emb_id}")
        if emb_id != 151697:
            raise RuntimeError(
                f"<|embodied_action|> got id {emb_id}, expected 151697 — token order "
                "does not match the training tokenizer; Step 3 would extract wrong positions."
            )
        proc.save_pretrained(out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not save processor: {type(e).__name__}: {e}")
        raise

    verify_qwen_ir(out_dir)


def verify_qwen_ir(out_dir: Path) -> None:
    """The language IR's hidden-state output is literally NAMED 'logits' (it is
    named positionally). Assert its last dim is the hidden size, not the vocab
    size, which is what distinguishes hidden states from real logits.
    """
    print("\n  --- Qwen3-VL IR check ---")
    core = ov.Core()
    lm_xml = out_dir / "openvino_language_model.xml"
    if not lm_xml.exists():
        raise RuntimeError(f"expected {lm_xml} to exist")

    m = core.read_model(lm_xml)
    for xml in sorted(out_dir.glob("*.xml")):
        mm = core.read_model(xml)
        names = [o.get_any_name() for o in mm.outputs]
        shown = [n for n in names if "present" not in n]
        print(f"    {xml.name}: {shown} (+{len(names) - len(shown)} present.*)")

    main = m.outputs[0]
    last_dim = main.get_partial_shape()[-1]
    print(f"    language model main output {main.get_any_name()!r} shape={main.get_partial_shape()}")
    if last_dim.is_dynamic or last_dim.get_length() != 2048:
        raise RuntimeError(
            f"language IR main output last dim is {last_dim}, expected 2048 (hidden size). "
            "This means logits were exported instead of hidden states — check that the bare "
            "Qwen3VLModel (not Qwen3VLForConditionalGeneration) was used."
        )
    print("    OK — output is last_hidden_state [.., .., 2048] (named 'logits' positionally)")

    kv_outs = [o.get_any_name() for o in m.outputs if "present" in o.get_any_name()]
    if kv_outs:
        raise RuntimeError(
            f"language IR exposes {len(kv_outs)} `present.*` KV outputs — expected 0 with "
            "stateful=True. VLA-JEPA is prefill-only and never consumes the KV cache."
        )
    print(f"    OK — {len(m.outputs)} output(s), no `present.*` KV tensors (stateful)")


# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description="Export VLA-JEPA to OpenVINO")
    p.add_argument("--checkpoint", default="pretrained/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt")
    p.add_argument("--config", default="pretrained/LIBERO/config.yaml")
    p.add_argument("--qwen-base", default="pretrained/Qwen3-VL-2B-Instruct")
    p.add_argument("--output-dir", default="openvino_model")
    p.add_argument("--golden-dir", default="golden_outputs")
    p.add_argument("--device", default="CPU", help="device used for validation")
    p.add_argument("--only", choices=["dit", "qwen"], default=None)
    p.add_argument("--int4", action="store_true", help="INT4 weight compression for Qwen3-VL")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.only in (None, "dit"):
        export_dit(args)
    if args.only in (None, "qwen"):
        export_qwen(args)

    golden_cfg = Path(args.golden_dir) / "config.json"
    if golden_cfg.exists():
        shutil.copy(golden_cfg, Path(args.output_dir) / "config.json")
        print(f"\nCopied {golden_cfg} -> {args.output_dir}/config.json")

    print("\nExport complete.")


if __name__ == "__main__":
    main()
