"""OpenVINO helper for ByteDance/Bernini-R-1.3B-Diffusers.

Bernini-R is a multi-task (t2i / i2i / t2v / v2v / r2v / rv2v) diffusion renderer
fine-tuned from Wan2.1-1.3B. It re-uses the Wan components: a custom
``WanTransformer3DModel`` (the DiT), a ``UMT5EncoderModel`` text encoder, an
``AutoencoderKLWan`` VAE and a ``UniPCMultistepScheduler``. The reference
implementation lives in the ``bernini`` python package
(https://github.com/bytedance/Bernini).

All of the data-dependent control flow (the per-timestep denoising loop, the 7
guidance modes, the source-id rotary-embedding construction, the token
assembly) lives in pure python inside ``bernini``'s
``BerniniRendererPipeline.__call__`` / ``GEN_Wanx22.sample``. OpenVINO static
graphs cannot express that control flow, so we only push the *leaf* compute into
OpenVINO and keep every loop / branch in python:

  * text encoder  -> a single static graph (token length fixed to 512);
  * transformer   -> ``BlocksCore``: condition-embedder + transformer blocks +
                     output projection, with the patch-embedding + rotary
                     construction kept in torch (``patch_vae_latent``). The
                     packed-token axis is dynamic so the same graph serves every
                     guidance combo (uncond / V / VI / VTI ...);
  * VAE encoder / decoder -> one static graph per *temporal* latent length
                     (the Wan VAE decodes frame-by-frame with a causal feature
                     cache, i.e. a python loop; tracing unrolls it for a fixed
                     length, so we compile lazily per length and cache).

The original ``bernini`` pipeline / sampler methods are then re-used verbatim by
injecting these OpenVINO-backed leaf modules, which guarantees the OV pipeline
stays numerically aligned with the reference across all tasks.
"""

import gc
import json
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import openvino as ov
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable


# --------------------------------------------------------------------------- #
# Saved artifact names
# --------------------------------------------------------------------------- #
TEXT_ENCODER_PATH = "text_encoder.xml"
TRANSFORMER_PATH = "transformer_blocks.xml"
VAE_DECODER_DIR = "vae_decoder"          # holds vae_decoder_t{T}.xml
VAE_ENCODER_DIR = "vae_encoder"          # holds vae_encoder_t{T}.xml
PATCH_EMBED_PATH = "patch_embedding.pt"  # tiny Conv3d weights kept in torch
TRANSFORMER_CONFIG_PATH = "transformer_config.json"
MAX_SEQ_LEN = 512


def cleanup_torchscript_cache():
    """Drop TorchScript caches between conversions (see the Wan helper)."""
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def repair_transformer_config(model_dir):
    """Reconcile ``transformer/config.json`` with the actual checkpoint shapes.

    Some mirrors of the Bernini-R snapshot ship a transformer ``config.json`` whose
    ``num_layers`` / ``num_attention_heads`` / ``ffn_dim`` do not match the weights
    (e.g. 40 / 40 / 13824 while the weights are 30 / 12 / 8960). ``from_pretrained``
    would then build a wrong-shaped model and fail to load. This rewrites the config
    in place from the checkpoint: ``num_layers`` from the block indices, ``ffn_dim``
    and ``inner_dim`` (hence ``num_attention_heads``) from tensor shapes.
    """
    import glob
    import re

    from safetensors import safe_open

    model_dir = Path(model_dir)
    cfg_path = model_dir / "transformer" / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    key_to_file = {}
    for shard in glob.glob(str(model_dir / "transformer" / "*.safetensors")):
        with safe_open(shard, framework="pt") as sf:
            for k in sf.keys():
                key_to_file[k] = shard
    if not key_to_file:
        return cfg  # weights not present yet; nothing to reconcile

    def shape_of(key):
        with safe_open(key_to_file[key], framework="pt") as sf:
            return list(sf.get_slice(key).get_shape())

    num_layers = 1 + max(
        int(m.group(1)) for k in key_to_file if (m := re.search(r"blocks\.(\d+)\.", k))
    )
    inner_dim = shape_of("blocks.0.attn1.to_q.weight")[0]
    ffn_dim = shape_of("blocks.0.ffn.net.0.proj.weight")[0]
    head_dim = cfg["attention_head_dim"]

    expected = {
        "num_layers": num_layers,
        "num_attention_heads": inner_dim // head_dim,
        "ffn_dim": ffn_dim,
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        print(f"⚠️ transformer/config.json mismatched checkpoint; repairing {expected}")
        cfg.update(expected)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return cfg


# --------------------------------------------------------------------------- #
# Real-valued rotary embedding (avoids complex ops in the OpenVINO graph)
# --------------------------------------------------------------------------- #
def _rotary_to_cos_sin(rotary_emb: torch.Tensor):
    """Convert bernini's complex rotary tensor into real ``(cos, sin)``.

    ``rotary_emb`` is the (concatenated) output of ``WanRotaryPosEmbed`` with
    shape ``[1, 1, L, head_dim // 2]`` and complex dtype. ``GEN_Wanx22.sample``
    feeds it to the transformer as-is (the model then does
    ``rotary_emb.transpose(1, 2)``); here we pre-transpose to ``[1, L, 1, hd/2]``
    and split into real/imaginary parts so the traced graph only sees real ops.
    """
    rotary_emb = rotary_emb.transpose(1, 2)  # [1, L, 1, hd/2], complex
    cos = torch.view_as_real(rotary_emb)[..., 0].contiguous()  # [1, L, 1, hd/2]
    sin = torch.view_as_real(rotary_emb)[..., 1].contiguous()
    return cos, sin


def _apply_rotary_real(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding with real arithmetic.

    ``x`` is ``[1, L, heads, head_dim]``; ``cos`` / ``sin`` are
    ``[1, L, 1, head_dim // 2]``. Mirrors the complex multiply used by bernini's
    ``_apply_rotary_emb`` (pairs of adjacent channels rotated by the angle whose
    cosine / sine are ``cos`` / ``sin``).
    """
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    out = torch.stack((out0, out1), dim=-1).flatten(-2)
    return out.type_as(x)


class _SDPAProcessor:
    """SDPA attention processor equivalent to ``WanAttnProcessor2_0``.

    On a single sample with Ulysses disabled, bernini's variable-length
    attention (``cu_seqlens = [0, L]``) reduces to ordinary full attention, so we
    replace it with ``F.scaled_dot_product_attention``. ``rotary_emb`` is passed
    as a real ``(cos, sin)`` tuple for self-attention and ``None`` for
    cross-attention. All sequence-parallel kwargs are accepted and ignored.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None, **kwargs):
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(kv_input)
        value = attn.to_v(kv_input)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))  # [1, Lq, H, D]
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            cos, sin = rotary_emb
            query = _apply_rotary_real(query, cos, sin)
            key = _apply_rotary_real(key, cos, sin)

        # [1, L, H, D] -> [1, H, L, D] for SDPA
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class BlocksCore(torch.nn.Module):
    """Static, dynamic-length core of ``WanTransformer3DModel``.

    Wraps the real ``condition_embedder``, transformer ``blocks`` (with their
    attention processors swapped for :class:`_SDPAProcessor`), ``norm_out`` and
    ``proj_out``. The patch-embedding and rotary construction are *not* part of
    this graph -- they stay in torch (see :class:`OVTransformerWrapper`).

    Forward inputs (all real-valued, batch size 1, dynamic token length ``L``):
        hidden_states         [1, L, inner_dim]   -- already patch-embedded
        timestep              [1]
        encoder_hidden_states [1, T_txt, text_dim]
        rope_cos / rope_sin   [1, L, 1, head_dim // 2]
    Returns the projected noise prediction ``[1, L, out_channels * prod(patch)]``.
    """

    def __init__(self, transformer):
        super().__init__()
        self.condition_embedder = transformer.condition_embedder
        self.blocks = transformer.blocks
        self.norm_out = transformer.norm_out
        self.proj_out = transformer.proj_out
        self.scale_shift_table = transformer.scale_shift_table
        for block in self.blocks:
            block.attn1.processor = _SDPAProcessor()
            block.attn2.processor = _SDPAProcessor()

    def forward(self, hidden_states, timestep, encoder_hidden_states, rope_cos, rope_sin):
        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep, encoder_hidden_states, None
        )
        # [1, 6*dim] -> [1, 6, dim]; broadcasts over tokens (single sample).
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        temb = temb.unsqueeze(1)  # [1, 1, dim] -> broadcasts over tokens

        rotary = (rope_cos, rope_sin)
        for block in self.blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary,
            )

        shift, scale = self.scale_shift_table.float().chunk(2, dim=1)  # [1,1,dim] each
        shift = shift + temb.float()
        scale = scale + temb.float()
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
            hidden_states
        )
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def _compress(ov_model, compression_config):
    if compression_config is not None:
        import nncf

        ov_model = nncf.compress_weights(ov_model, **compression_config)
    return ov_model


def convert_pipeline(model_dir, output_dir, compression_config=None, vae_latent_frames=(1,)):
    """Convert Bernini-R to OpenVINO IR.

    Args:
        model_dir: local path to the Bernini-R-1.3B-Diffusers snapshot.
        output_dir: where the IR / artifacts are written.
        compression_config: ``None`` (FP16) or kwargs for ``nncf.compress_weights``.
        vae_latent_frames: temporal latent lengths to pre-convert VAE graphs for
            (``1`` covers images / single-frame; video lengths are also compiled
            lazily at run time).
    """
    from bernini.pipeline import BerniniRendererPipeline

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    done = (
        (output_dir / TEXT_ENCODER_PATH).exists()
        and (output_dir / TRANSFORMER_PATH).exists()
        and (output_dir / PATCH_EMBED_PATH).exists()
        and all((output_dir / VAE_DECODER_DIR / f"vae_decoder_t{t}.xml").exists() for t in vae_latent_frames)
        and all((output_dir / VAE_ENCODER_DIR / f"vae_encoder_t{t}.xml").exists() for t in vae_latent_frames)
    )
    if done:
        print(f"✅ model already converted; results in {output_dir}")
        return

    # Reconcile a possibly-mismatched transformer config with the checkpoint.
    repair_transformer_config(model_dir)

    print("⌛ loading the original Bernini-R pipeline (this loads ~6 GB of weights) ...")
    # ``wan22_base`` in config.json points at a HF repo id; override it to the
    # local snapshot so tokenizer / scheduler / sub-models load offline.
    pipe = BerniniRendererPipeline.from_pretrained(
        model_dir, device="cpu", load_ckpt_weights=False, wan22_base=str(model_dir)
    )
    transformer = pipe.model.diff_dec.transformer
    transformer.eval()
    text_encoder = pipe.model.t5_text_encoder
    text_encoder.eval()
    vae = pipe.vae
    vae.eval()

    # tokenizer + scheduler are plain artifacts (kept in python at run time).
    pipe.tokenizer.save_pretrained(output_dir / "tokenizer")
    pipe.model.diff_dec.scheduler.save_pretrained(output_dir / "scheduler")
    with open(output_dir / TRANSFORMER_CONFIG_PATH, "w") as f:
        json.dump(dict(transformer.config), f)

    # ---- transformer: patch-embedding (torch) + BlocksCore (OpenVINO) ----
    if not (output_dir / PATCH_EMBED_PATH).exists():
        torch.save(transformer.patch_embedding.state_dict(), output_dir / PATCH_EMBED_PATH)

    if not (output_dir / TRANSFORMER_PATH).exists():
        print("⌛ converting transformer (BlocksCore) ...")
        inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        head_dim = transformer.config.attention_head_dim
        text_dim = transformer.config.text_dim
        L = 880  # representative packed-token length; axis stays dynamic
        blocks_core = BlocksCore(transformer).float().eval()
        example = {
            "hidden_states": torch.ones([1, L, inner_dim]),
            "timestep": torch.tensor([1000.0]),
            "encoder_hidden_states": torch.ones([1, MAX_SEQ_LEN, text_dim]),
            "rope_cos": torch.ones([1, L, 1, head_dim // 2]),
            "rope_sin": torch.zeros([1, L, 1, head_dim // 2]),
        }
        __make_16bit_traceable(blocks_core)
        ts = TorchScriptPythonDecoder(blocks_core, example_input=example, trace_kwargs={"check_trace": False})
        with torch.no_grad():
            ov_model = ov.convert_model(ts, example_input=example)
        ov_model = _compress(ov_model, compression_config)
        ov.save_model(ov_model, output_dir / TRANSFORMER_PATH)
        del ov_model, blocks_core
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ transformer converted")

    # ---- text encoder ----
    if not (output_dir / TEXT_ENCODER_PATH).exists():
        print("⌛ converting text encoder (UMT5) ...")

        class _TE(torch.nn.Module):
            def __init__(self, te):
                super().__init__()
                self.te = te

            def forward(self, input_ids, attention_mask):
                return self.te(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        te_wrap = _TE(text_encoder).eval()
        example = {
            "input_ids": torch.ones([1, MAX_SEQ_LEN], dtype=torch.long),
            "attention_mask": torch.ones([1, MAX_SEQ_LEN], dtype=torch.long),
        }
        __make_16bit_traceable(te_wrap)
        ts = TorchScriptPythonDecoder(te_wrap, example_input=example, trace_kwargs={"check_trace": False})
        with torch.no_grad():
            ov_model = ov.convert_model(ts, example_input=example)
        ov_model = _compress(ov_model, compression_config)
        ov.save_model(ov_model, output_dir / TEXT_ENCODER_PATH)
        del ov_model, te_wrap
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ text encoder converted")

    # ---- VAE decoder / encoder, one graph per temporal latent length ----
    for t in vae_latent_frames:
        _convert_vae_decoder(vae, t, output_dir, compression_config)
        _convert_vae_encoder(vae, t, output_dir, compression_config)

    del pipe, transformer, text_encoder, vae
    gc.collect()
    print(f"✅ conversion finished; results in {output_dir}")


def _fix_vae_upsample(vae):
    """Replace ``nearest-exact`` upsampling with ``nearest``.

    OpenVINO has no ``aten::_upsample_nearest_exact2d`` translation, so we switch
    the Wan VAE's ``WanUpsample`` modules to plain ``nearest`` (matching the Wan
    notebook's upsample fix). The numerical difference is negligible for the 2x
    integer upsampling used here.
    """
    for module in vae.modules():
        if isinstance(module, torch.nn.Upsample) and module.mode == "nearest-exact":
            module.mode = "nearest"


class _VAEDecodeWrap(torch.nn.Module):
    """Wraps ``AutoencoderKLWan.decode`` to a plain tensor forward for tracing."""

    def __init__(self, vae):
        super().__init__()
        _fix_vae_upsample(vae)
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class _VAEEncodeWrap(torch.nn.Module):
    """Wraps ``AutoencoderKLWan.encode`` returning the deterministic mode."""

    def __init__(self, vae):
        super().__init__()
        _fix_vae_upsample(vae)
        self.vae = vae

    def forward(self, x):
        return self.vae.encode(x, return_dict=False)[0].mode()


def _convert_vae_decoder(vae, t_latent, output_dir, compression_config):
    out = output_dir / VAE_DECODER_DIR / f"vae_decoder_t{t_latent}.xml"
    if out.exists():
        return
    print(f"⌛ converting VAE decoder (T_latent={t_latent}) ...")
    out.parent.mkdir(parents=True, exist_ok=True)
    z = torch.ones([1, vae.config.z_dim, t_latent, 60, 104])
    wrap = _VAEDecodeWrap(vae).eval()
    with torch.no_grad():
        ov_model = ov.convert_model(wrap, example_input=z)
    ov_model = _compress(ov_model, compression_config)
    ov.save_model(ov_model, out)
    del ov_model, wrap
    cleanup_torchscript_cache()
    gc.collect()
    print(f"✅ VAE decoder (T_latent={t_latent}) converted")


def _convert_vae_encoder(vae, t_latent, output_dir, compression_config):
    out = output_dir / VAE_ENCODER_DIR / f"vae_encoder_t{t_latent}.xml"
    if out.exists():
        return
    # pixel frames that map to ``t_latent`` latent frames: T_pix = (t-1)*4 + 1
    t_pix = (t_latent - 1) * 4 + 1
    print(f"⌛ converting VAE encoder (T_pixels={t_pix}) ...")
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.ones([1, 3, t_pix, 480, 832])
    wrap = _VAEEncodeWrap(vae).eval()
    with torch.no_grad():
        ov_model = ov.convert_model(wrap, example_input=x)
    ov_model = _compress(ov_model, compression_config)
    ov.save_model(ov_model, out)
    del ov_model, wrap
    cleanup_torchscript_cache()
    gc.collect()
    print(f"✅ VAE encoder (T_pixels={t_pix}) converted")


# --------------------------------------------------------------------------- #
# Run-time wrappers
# --------------------------------------------------------------------------- #
core = ov.Core()


# Per-component GPU/NPU precision. The UMT5 text encoder is prone to fp16
# overflow (a well-known T5/UMT5 issue): for some prompts its fp16 activations
# blow up, the cross-attention conditioning becomes garbage and the decoded image
# collapses to black. The Wan VAE decoder (large ``WanRMS_norm`` scales) is also
# overflow-prone. Both run once per generation, so forcing them to fp32 costs
# almost nothing, while the expensive transformer (run several times per step,
# every step) stays in fast fp16. This keeps the GPU fast and the output stable.
DEFAULT_GPU_CONFIG = {
    "transformer": {"INFERENCE_PRECISION_HINT": "f16"},
    "text_encoder": {"INFERENCE_PRECISION_HINT": "f32"},
    "vae": {"INFERENCE_PRECISION_HINT": "f32"},
}


def _device_config(device, ov_config, component=None):
    """Per-device, per-component OpenVINO config.

    On GPU/NPU we keep the transformer / text encoder in fast fp16 and force the
    VAE to fp32 (see ``DEFAULT_GPU_CONFIG``) to avoid the fp16 overflow that turns
    the decoded image black. ``ov_config`` (a flat dict applied to every
    component) overrides the defaults.
    """
    cfg = {}
    dev = (device or "").upper()
    if "GPU" in dev or "NPU" in dev:
        cfg.update(DEFAULT_GPU_CONFIG.get(component, {}))
    if ov_config:
        cfg.update(ov_config)
    return cfg


class _Out:
    """Tiny stand-in exposing ``.sample`` / ``.last_hidden_state`` / ``[0]``."""

    def __init__(self, tensor):
        self.sample = tensor
        self.last_hidden_state = tensor

    def __getitem__(self, idx):
        return self.sample


class OVTextEncoderWrapper(torch.nn.Module):
    def __init__(self, model_dir, device, ov_config=None):
        super().__init__()
        self._model = core.compile_model(Path(model_dir) / TEXT_ENCODER_PATH, device,
                                          _device_config(device, ov_config, "text_encoder"))
        self.dtype = torch.float32

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        res = self._model([input_ids.to(torch.long), attention_mask.to(torch.long)])[0]
        return _Out(torch.from_numpy(res))

    __call__ = forward

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


class OVTransformerWrapper(torch.nn.Module):
    """OpenVINO-backed ``WanTransformer3DModel`` stand-in.

    Keeps ``patch_embedding`` (Conv3d) and ``rope`` (no params) in torch so that
    ``patch_vae_latent`` -- called many times per step by ``GEN_Wanx22.sample``
    with different source ids -- runs exactly as in the reference. The heavy
    block stack runs in OpenVINO via :class:`BlocksCore`.
    """

    def __init__(self, model_dir, device, ov_config=None):
        super().__init__()
        from bernini.models.transformer_wan import WanRotaryPosEmbed
        from diffusers.configuration_utils import FrozenDict

        model_dir = Path(model_dir)
        with open(model_dir / TRANSFORMER_CONFIG_PATH) as f:
            cfg = json.load(f)
        self.config = FrozenDict(cfg)
        self.dtype = torch.float32

        inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        patch = tuple(cfg["patch_size"])
        self.patch_embedding = torch.nn.Conv3d(cfg["in_channels"], inner_dim, kernel_size=patch, stride=patch)
        self.patch_embedding.load_state_dict(torch.load(model_dir / PATCH_EMBED_PATH, map_location="cpu"))
        self.patch_embedding.eval()
        self.rope = WanRotaryPosEmbed(
            cfg["attention_head_dim"], patch, cfg["rope_max_seq_len"],
            use_src_id_rotary_emb=cfg["use_src_id_rotary_emb"],
        )
        self._model = core.compile_model(model_dir / TRANSFORMER_PATH, device,
                                          _device_config(device, ov_config, "transformer"))

    def patch_vae_latent(self, hidden_states, source_id=None):
        """Patch-embed a VAE latent ``[B,C,T,H,W]`` -> tokens + complex rotary."""
        hidden_states = hidden_states.to(torch.float32)
        rotary_emb = self.rope(hidden_states, source_id)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states, rotary_emb

    def forward(self, hidden_states, timestep, encoder_hidden_states=None,
                rotary_emb=None, batch_image_vae_seqlen=None, text_features_length=None,
                return_dict=True):
        cos, sin = _rotary_to_cos_sin(rotary_emb)
        res = self._model([
            hidden_states.to(torch.float32),
            timestep.to(torch.float32),
            encoder_hidden_states.to(torch.float32),
            cos.to(torch.float32),
            sin.to(torch.float32),
        ])[0]
        return _Out(torch.from_numpy(res))

    __call__ = forward

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


class _LatentDist:
    def __init__(self, tensor):
        self._t = tensor

    def mode(self):
        return self._t

    def sample(self, generator=None):
        return self._t


class OVVAEWrapper(torch.nn.Module):
    """OpenVINO-backed ``AutoencoderKLWan`` stand-in.

    The Wan VAE decodes / encodes the temporal axis with a python loop + causal
    feature cache, so a traced graph is valid for one temporal length only. We
    compile lazily per length and cache the compiled models, which keeps the
    OpenVINO graphs free of data-dependent loops.
    """

    def __init__(self, ov_model_dir, device, ov_config=None, original_config=None,
                 source_model_dir=None, compression_config=None):
        super().__init__()
        from diffusers.configuration_utils import FrozenDict

        self._ov_dir = Path(ov_model_dir)
        self._source_model_dir = Path(source_model_dir) if source_model_dir else None
        self._compression_config = compression_config
        self._device = device
        self._ov_config = _device_config(device, ov_config, "vae")
        self.config = FrozenDict(original_config)
        self.temperal_downsample = list(original_config["temperal_downsample"])
        self.dtype = torch.float32
        self._dec_cache = {}
        self._enc_cache = {}
        self._torch_vae = None  # loaded lazily only if a new length must be converted

    def _ensure_torch_vae(self):
        if self._torch_vae is None:
            if self._source_model_dir is None:
                raise RuntimeError(
                    "A VAE graph for this temporal length was not pre-converted and the "
                    "original model dir is unavailable for on-demand conversion. Re-run "
                    "convert_pipeline with the required vae_latent_frames."
                )
            from diffusers.models import AutoencoderKLWan

            print("⌛ loading original VAE for on-demand graph conversion ...")
            self._torch_vae = AutoencoderKLWan.from_pretrained(
                self._source_model_dir, subfolder="vae", torch_dtype=torch.float32
            ).eval()
        return self._torch_vae

    def _decoder(self, t_latent):
        if t_latent not in self._dec_cache:
            xml = self._ov_dir / VAE_DECODER_DIR / f"vae_decoder_t{t_latent}.xml"
            if not xml.exists():
                print(f"⌛ converting VAE decoder on demand (T_latent={t_latent}) ...")
                _convert_vae_decoder(self._ensure_torch_vae(), t_latent, self._ov_dir,
                                     self._compression_config)
            self._dec_cache[t_latent] = core.compile_model(xml, self._device, self._ov_config)
        return self._dec_cache[t_latent]

    def _encoder(self, t_pix):
        # map pixel frames -> latent frames -> graph key
        t_latent = (t_pix - 1) // 4 + 1
        if t_latent not in self._enc_cache:
            xml = self._ov_dir / VAE_ENCODER_DIR / f"vae_encoder_t{t_latent}.xml"
            if not xml.exists():
                print(f"⌛ converting VAE encoder on demand (T_latent={t_latent}) ...")
                _convert_vae_encoder(self._ensure_torch_vae(), t_latent, self._ov_dir,
                                     self._compression_config)
            self._enc_cache[t_latent] = core.compile_model(xml, self._device, self._ov_config)
        return self._enc_cache[t_latent]

    def decode(self, z, return_dict=True):
        t_latent = z.shape[2]
        res = self._decoder(t_latent)([z.to(torch.float32)])[0]
        out = torch.from_numpy(res)
        if not return_dict:
            return (out,)
        return _Out(out)

    def encode(self, x, return_dict=True):
        res = self._encoder(x.shape[2])([x.to(torch.float32)])[0]
        dist = _LatentDist(torch.from_numpy(res))
        if not return_dict:
            return (dist,)
        from diffusers.models.modeling_outputs import AutoencoderKLOutput

        return AutoencoderKLOutput(latent_dist=dist)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


# --------------------------------------------------------------------------- #
# Assembled pipeline (re-uses bernini's sample / __call__ verbatim)
# --------------------------------------------------------------------------- #
def load_ov_pipeline(model_dir, ov_model_dir, device_map="CPU", ov_config=None,
                     compression_config=None):
    """Build a ``BerniniRendererPipeline`` whose leaf modules run on OpenVINO.

    ``device_map`` is a string or a dict with keys ``transformer`` /
    ``text_encoder`` / ``vae``. All of the denoising / guidance / VAE
    orchestration is the original ``bernini`` code -- only the three leaf modules
    are replaced, so behaviour matches the reference across every task.

    ``compression_config`` (the same one used for ``convert_pipeline``) is reused
    when a VAE graph for a new video length has to be converted on demand.
    """
    from transformers import AutoTokenizer
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.wan_diffusion import GEN_Wanx22
    from bernini.pipeline import BerniniRendererPipeline

    model_dir = Path(model_dir)
    ov_model_dir = Path(ov_model_dir)
    if isinstance(device_map, str):
        device_map = {"transformer": device_map, "text_encoder": device_map, "vae": device_map}

    config = BerniniRendererConfig.from_pretrained(model_dir)

    with open(model_dir / "vae" / "config.json") as f:
        vae_config = json.load(f)

    # OpenVINO-backed leaf modules.
    ov_text_encoder = OVTextEncoderWrapper(ov_model_dir, device_map["text_encoder"], ov_config)
    ov_transformer = OVTransformerWrapper(ov_model_dir, device_map["transformer"], ov_config)
    ov_vae = OVVAEWrapper(ov_model_dir, device_map["vae"], ov_config, original_config=vae_config,
                          source_model_dir=model_dir, compression_config=compression_config)

    # Diffusion decoder -- re-use GEN_Wanx22's methods, skip its heavy __init__.
    diff_dec = GEN_Wanx22.__new__(GEN_Wanx22)
    torch.nn.Module.__init__(diff_dec)
    diff_dec.config = config
    diff_dec.switch_dit_boundary = config.switch_dit_boundary
    diff_dec.model_id_or_path = str(model_dir)
    diff_dec.transformer = ov_transformer
    diff_dec.transformer_2 = None
    diff_dec.rope = ov_transformer.rope
    diff_dec.use_unipc = config.use_unipc
    diff_dec.scheduler = UniPCMultistepScheduler.from_pretrained(
        ov_model_dir / "scheduler", flow_shift=config.shift
    )
    diff_dec.vae_scale_factor_temporal = 4
    diff_dec.vae_scale_factor_spatial = 8

    # Renderer model -- re-use BerniniRendererModel.sample / encode_prompt.
    model = BerniniRendererModel.__new__(BerniniRendererModel)
    torch.nn.Module.__init__(model)
    model.config = config
    model.max_sequence_length = config.max_sequence_length
    model.t5_text_encoder = ov_text_encoder
    model.diff_dec = diff_dec

    tokenizer = AutoTokenizer.from_pretrained(ov_model_dir / "tokenizer", trust_remote_code=True)
    pipe = BerniniRendererPipeline(model=model, vae=ov_vae, tokenizer=tokenizer, device="cpu")
    return pipe


# Guidance mode + system prompt + required inputs per task, matching the
# reference testcases in github.com/bytedance/Bernini (assets/testcases/*.json
# and bernini.prompt_enhancer.SYSTEM_PROMPTS). The system prompt is prefixed to
# the user prompt by BerniniRendererPipeline.__call__.
TASK_GUIDANCE = {
    "t2i": "t2v_apg",   # text -> image (single frame)
    "t2v": "t2v_apg",   # text -> video
    "i2i": "v2v",       # image edit (reference image -> image)
    "v2v": "v2v_apg",   # video edit (source video -> video)
    "r2v": "r2v_apg",   # reference image(s) -> video
    "rv2v": "rv2v",     # reference image(s) + source video -> video
}

TASK_SYSTEM_PROMPT = {
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
}

# Which visual inputs each task consumes (drives the demo UI + notebook calls).
TASK_INPUTS = {
    "t2i": (),
    "t2v": (),
    "i2i": ("image",),
    "v2v": ("video",),
    "r2v": ("images",),
    "rv2v": ("video", "images"),
}
