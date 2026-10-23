"""Static-shape PyTorch wrappers and pre/post-processing for Cosmos3-Edge text-to-video with OpenVINO."""

import json
import re
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import torch.nn.functional as F
from openvino import opset13 as ops
from torch import nn

NUM_FRAMES, HEIGHT, WIDTH, FPS = 17, 256, 256, 20.0
TEMPORAL_COMPRESSION = 4
if (NUM_FRAMES - 1) % TEMPORAL_COMPRESSION:
    raise ValueError("NUM_FRAMES must be 1 + a multiple of the temporal compression factor")
LATENT_FRAMES = (NUM_FRAMES - 1) // TEMPORAL_COMPRESSION + 1
LATENT_SHAPE = (1, 48, LATENT_FRAMES, 16, 16)
TEXT_INPUT_NAMES = ("input_ids", "text_cos", "text_sin")
DENOISER_INPUT_NAMES = ("vision_latents", "timestep", "text_k", "text_v", "text_mask", "gen_cos", "gen_sin")


class PromptPreprocessor:
    """Tokenizes the conditional and unconditional prompts and builds the static text and 3D mRoPE tables."""

    def __init__(self, checkpoint: str | Path, lmax: int = 256, num_frames: int = NUM_FRAMES, height: int = HEIGHT, width: int = WIDTH, fps: float = FPS):
        from transformers import AutoTokenizer

        checkpoint = Path(checkpoint)
        self.lmax, self.num_frames, self.height, self.width, self.fps = lmax, num_frames, height, width, fps
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint / "text_tokenizer", local_files_only=True)
        config = json.loads((checkpoint / "transformer" / "config.json").read_text())
        self.head_dim = config["head_dim"]
        self.rope_axes_dim = config.get("rope_axes_dim", config["rope_scaling"]["mrope_section"])
        self.temporal_margin = config["unified_3d_mrope_temporal_modality_margin"]
        self.base_fps = config["base_fps"]
        self.temporal_compression = config.get("temporal_compression_factor", 4)
        self.patch_size = config["latent_patch_size"]
        self.eos_token_id = self.tokenizer.eos_token_id
        self.vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.inv_freq = 1.0 / (config["rope_theta"] ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))

    def _tokenize(self, text):
        encoding = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=True,
        )
        return list(encoding.input_ids) + [self.eos_token_id, self.vision_start_id]

    def _rotary(self, position_ids):
        frequencies = torch.as_tensor(position_ids, dtype=torch.float32)[:, :, None] * self.inv_freq[None, None, :]
        interleaved = frequencies[0].clone()
        for axis, offset in enumerate((1, 2), start=1):
            length = self.rope_axes_dim[axis] * 3
            interleaved[..., slice(offset, length, 3)] = frequencies[axis, ..., slice(offset, length, 3)]
        embedding = torch.cat((interleaved, interleaved), dim=-1)
        return embedding.cos().numpy(), embedding.sin().numpy()

    def _vision_positions(self, text_length):
        latent_t, latent_h, latent_w = LATENT_SHAPE[2:]
        grid_h, grid_w = latent_h // self.patch_size, latent_w // self.patch_size
        temporal = (
            torch.arange(latent_t, dtype=torch.float32) / (self.fps / self.temporal_compression) * (self.base_fps / self.temporal_compression)
            + text_length
            + self.temporal_margin
        )
        temporal = temporal[:, None].expand(-1, grid_h * grid_w).flatten()
        rows = torch.arange(grid_h, dtype=torch.float32)[None, :, None].expand(latent_t, -1, grid_w).flatten()
        cols = torch.arange(grid_w, dtype=torch.float32)[None, None, :].expand(latent_t, grid_h, -1).flatten()
        return torch.stack((temporal, rows, cols))

    def __call__(self, prompt: str, negative_prompt: str | None = None) -> dict:
        duration = self.num_frames / self.fps
        conditional = (
            f"{prompt.rstrip('.')}. The video is {duration:.1f} seconds long and is of "
            f"{self.fps:.0f} FPS. This video is of {self.height}x{self.width} resolution."
        )
        negative = (negative_prompt or "").rstrip(".")
        unconditional = (
            f"{negative + '. ' if negative else ''}The video is not {duration:.1f} seconds long and is not of "
            f"{self.fps:.0f} FPS. This video is not of {self.height}x{self.width} resolution."
        )
        rows = [self._tokenize(conditional), self._tokenize(unconditional)]
        lengths = [len(row) for row in rows]
        if max(lengths) > self.lmax:
            raise ValueError(f"Prompt needs {max(lengths)} tokens; the model supports {self.lmax}")

        input_ids = np.zeros((2, self.lmax), dtype=np.int64)
        text_cos = np.ones((2, self.lmax, self.head_dim), dtype=np.float32)
        text_sin = np.zeros((2, self.lmax, self.head_dim), dtype=np.float32)
        text_mask = np.full((2, 1, 1, self.lmax), -10000.0, dtype=np.float32)
        gen_cos, gen_sin = [], []
        for row, (tokens, length) in enumerate(zip(rows, lengths)):
            input_ids[row, :length] = tokens
            text_cos[row, :length], text_sin[row, :length] = self._rotary(torch.arange(length, dtype=torch.float32).expand(3, -1))
            text_mask[row, :, :, :length] = 0
            cosine, sine = self._rotary(self._vision_positions(length))
            gen_cos.append(cosine[:, None])
            gen_sin.append(sine[:, None])
        return {
            "input_ids": input_ids,
            "text_cos": text_cos,
            "text_sin": text_sin,
            "text_mask": text_mask,
            "gen_cos": np.concatenate(gen_cos),
            "gen_sin": np.concatenate(gen_sin),
            "token_lengths": lengths,
        }


def make_scheduler(config: dict, steps: int):
    """UniPC scheduler with the native flow sigma schedule used by `Cosmos3OmniPipeline`."""
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_config(config)
    scheduler.set_timesteps(steps, sigmas=np.linspace(1.0 - 1.0 / scheduler.config.num_train_timesteps, 0.0, steps + 1)[:-1])
    return scheduler


def run_denoising(step_fn, latent: np.ndarray, scheduler_config: dict, steps: int) -> np.ndarray:
    """
    Denoise a latent with UniPC.

    :param step_fn: maps (latent, timestep) as float32 arrays to the guided velocity
    :param latent: initial noise of shape `LATENT_SHAPE`
    :param scheduler_config: UniPC scheduler configuration
    :param steps: number of denoising steps
    :returns: denoised latent
    """
    scheduler = make_scheduler(config=scheduler_config, steps=steps)
    x = torch.from_numpy(latent)
    for t in scheduler.timesteps:
        velocity = torch.from_numpy(np.asarray(step_fn(x.numpy(), np.array([float(t)], dtype=np.float32))))
        x = scheduler.step(velocity, t, x, return_dict=False)[0]
    return x.numpy()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], -1)


class TextKV(nn.Module):
    """Causal text pathway for the conditional and unconditional rows; returns per-layer RoPE'd K and V."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.tr = transformer

    def forward(self, input_ids, text_cos, text_sin):
        tr = self.tr
        batch, length = input_ids.shape
        cos, sin = text_cos[:, :, None], text_sin[:, :, None]
        hidden = tr.embed_tokens(input_ids)
        keys, values = [], []
        for index, layer in enumerate(tr.layers):
            attn = layer.self_attn
            normed = layer.input_layernorm(hidden)
            k = attn.norm_k(attn.to_k(normed).view(batch, length, attn.num_key_value_heads, attn.head_dim))
            k_gen = attn.k_norm_und_for_gen(k) if attn.k_norm_und_for_gen is not None else k
            v = attn.to_v(normed).view(batch, length, attn.num_key_value_heads, attn.head_dim)
            keys.append((k_gen * cos + rotate_half(k_gen) * sin).transpose(1, 2))
            values.append(v.transpose(1, 2))
            if index == len(tr.layers) - 1:
                break
            q = attn.norm_q(attn.to_q(normed).view(batch, length, attn.num_attention_heads, attn.head_dim))
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=True)
            hidden = hidden + attn.to_out(out.transpose(1, 2).reshape(batch, length, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return torch.stack(keys), torch.stack(values)


class DenoiseStep(nn.Module):
    """One denoising step with classifier-free guidance; cond and uncond rows share one pass through the linears."""

    def __init__(self, transformer: nn.Module, guidance_scale: float):
        super().__init__()
        self.tr = transformer
        self.guidance_scale = guidance_scale
        latent_t, latent_h, latent_w = LATENT_SHAPE[2:]
        patch = transformer.config.latent_patch_size
        self.shapes = [(latent_t, latent_h // patch, latent_w // patch)]
        self.noisy = [torch.arange(latent_t)]
        self.num_tokens = latent_t * (latent_h // patch) * (latent_w // patch)

    def forward(self, x, timestep, text_k, text_v, text_mask, gen_cos, gen_sin):
        tr, n = self.tr, self.num_tokens
        vision, original = tr._patchify_and_pack_latents([x])
        tokens = tr.proj_in(vision) + tr.time_embedder(tr.time_proj(timestep * tr.config.timestep_scale))
        hidden = torch.cat([tokens, tokens], 0)
        zeros = torch.zeros(1, 1, 1, n, dtype=text_mask.dtype)
        masks = [torch.cat([text_mask[row : row + 1], zeros], -1) for row in range(2)]
        for index, layer in enumerate(tr.layers):
            attn = layer.self_attn
            normed = layer.input_layernorm_moe_gen(hidden)
            q = attn.norm_added_q(attn.add_q_proj(normed).view(-1, attn.num_attention_heads, attn.head_dim))
            k = attn.norm_added_k(attn.add_k_proj(normed).view(-1, attn.num_key_value_heads, attn.head_dim))
            v = attn.add_v_proj(normed).view(-1, attn.num_key_value_heads, attn.head_dim)
            q = q * gen_cos + rotate_half(q) * gen_sin
            k = k * gen_cos + rotate_half(k) * gen_sin
            q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            outputs = []
            for row in range(2):
                rows = slice(row * n, (row + 1) * n)
                out = F.scaled_dot_product_attention(
                    q[:, :, rows],
                    torch.cat([text_k[index, row : row + 1], k[:, :, rows]], 2),
                    torch.cat([text_v[index, row : row + 1], v[:, :, rows]], 2),
                    attn_mask=masks[row],
                    enable_gqa=True,
                )
                outputs.append(out[0].transpose(0, 1).reshape(n, -1))
            hidden = hidden + attn.to_add_out(torch.cat(outputs, 0))
            hidden = hidden + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(hidden))
        projected = tr.proj_out(tr.norm_moe_gen(hidden))
        conditional = tr._unpatchify_and_unpack_latents(projected[:n], self.shapes, self.noisy, original)[0]
        unconditional = tr._unpatchify_and_unpack_latents(projected[n:], self.shapes, self.noisy, original)[0]
        return unconditional + self.guidance_scale * (conditional - unconditional)


class VAEDecoder(nn.Module):
    """Latent -> video wrapper around the Wan VAE decoder."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


def use_nearest_upsampling(model: nn.Module) -> int:
    """Switch 'nearest-exact' upsampling to 'nearest', which is identical for the integer scale factors used here."""
    upsamples = [m for m in model.modules() if isinstance(m, nn.Upsample) and m.mode == "nearest-exact"]
    for module in upsamples:
        module.mode = "nearest"
    return len(upsamples)


def _const(output):
    node = output.get_node()
    return np.array(node.get_data()) if node.get_type_name() == "Constant" else None


def _const_i64(values):
    return ops.constant(np.array(values, dtype=np.int64))


def sdpa_to_rank4(model: ov.Model) -> int:
    """Fold grouped-query 5D attention [B,G,R,L,D] x [B,G,1,S,D] into standard 4D attention with G*R heads."""
    count = 0
    for sdpa in model.get_ordered_ops():
        if sdpa.get_type_name() != "ScaledDotProductAttention" or len(sdpa.get_input_shape(0)) != 5:
            continue
        batch, groups, repeat, length, dim = sdpa.get_input_shape(0)
        k_batch, k_groups, k_repeat, source, _ = sdpa.get_input_shape(1)
        if k_repeat != 1 or k_groups != groups:
            continue
        heads = groups * repeat
        query = ops.reshape(sdpa.input_value(0), _const_i64([batch, heads, length, dim]), False)
        key, value = (
            ops.reshape(
                ops.broadcast(sdpa.input_value(i), _const_i64([k_batch, groups, repeat, source, dim])), _const_i64([k_batch, heads, source, dim]), False
            )
            for i in (1, 2)
        )
        extra = []
        if sdpa.get_input_size() > 3:
            mask_shape = list(sdpa.get_input_shape(3))
            if len(mask_shape) != 5 or mask_shape[1] != 1 or mask_shape[2] != 1:
                continue
            extra.append(ops.reshape(sdpa.input_value(3), _const_i64(mask_shape[:2] + mask_shape[3:]), False))
        if sdpa.get_input_size() > 4:
            extra.append(sdpa.input_value(4))
        attention = ops.scaled_dot_product_attention(query, key, value, *extra, causal=sdpa.get_attributes()["causal"])
        output = ops.reshape(attention, _const_i64([batch, groups, repeat, length, dim]), False)
        for target in list(sdpa.output(0).get_target_inputs()):
            target.replace_source_output(output.output(0))
        count += 1
    return count


F16_MAX = 65504.0


def _skip_converts(output):
    while output.get_node().get_type_name() == "Convert":
        output = output.get_node().input_value(0)
    return output


def _single_consumer(node, type_name):
    targets = list(node.output(0).get_target_inputs())
    if len(targets) != 1 or targets[0].get_node().get_type_name() != type_name:
        return None
    return targets[0].get_node()


def _perm(node):
    value = _const(node.input_value(1))
    return None if value is None else value.tolist()


def rmsnorm_f16(model):
    """Keep only the [1,1,T,H,W] reciprocal of WanRMSNorm in f32 instead of the full-size Divide and Converts."""
    count = 0
    for div in model.get_ordered_ops():
        if div.get_type_name() != "Divide":
            continue
        to_f32, clamp = div.input_value(0).get_node(), div.input_value(1).get_node()
        if to_f32.get_type_name() != "Convert" or clamp.get_type_name() != "Maximum" or to_f32.get_input_element_type(0) != ov.Type.f16:
            continue
        reduce, eps = clamp.input_value(0).get_node(), _const(clamp.input_value(1))
        if reduce.get_type_name() != "ReduceL2" or eps is None:
            continue
        to_f16 = _single_consumer(div, "Convert")
        if to_f16 is None or to_f16.get_output_element_type(0) != ov.Type.f16:
            continue
        x = to_f32.input_value(0)
        norm = ops.convert(ops.reduce_l2(x, reduce.input_value(1), True), ov.Type.f32)
        recip = ops.divide(ops.constant(np.ones([1] * eps.ndim, np.float32)), ops.maximum(norm, ops.constant(eps.astype(np.float32))))
        recip = ops.minimum(recip, ops.constant(np.full([1] * eps.ndim, F16_MAX, np.float32)))
        out = ops.multiply(x, ops.convert(recip, ov.Type.f16))
        for target in list(to_f16.output(0).get_target_inputs()):
            target.replace_source_output(out.output(0))
        count += 1
    return count


def upsampler_5d(model):
    """Resample: Transpose -> Reshape(frames) -> Interpolate -> Conv2D -> Reshape -> Transpose  =>  Interpolate(H,W) -> Conv3D(1x3x3)."""
    count = 0
    for conv in model.get_ordered_ops():
        if conv.get_type_name() != "Convolution" or len(conv.get_output_shape(0)) != 4:
            continue
        interp = _skip_converts(conv.input_value(0)).get_node()
        if interp.get_type_name() != "Interpolate" or interp.get_attributes().get("mode") != "nearest":
            continue
        frames = _skip_converts(interp.input_value(0)).get_node()
        if frames.get_type_name() != "Reshape":
            continue
        to_frames = frames.input_value(0).get_node()
        if to_frames.get_type_name() != "Transpose" or _perm(to_frames) != [0, 2, 1, 3, 4]:
            continue
        x5 = to_frames.input_value(0)
        bias = _single_consumer(conv, "Add")
        if x5.get_shape()[0] != 1 or bias is None or _const(bias.input_value(1)) is None:
            continue
        back = _single_consumer(bias, "Reshape")
        to_5d = _single_consumer(back, "Transpose") if back is not None else None
        weight = _const(_skip_converts(conv.input_value(1)))
        if to_5d is None or _perm(to_5d) != [0, 2, 1, 3, 4] or weight is None:
            continue
        attrs, conv_attrs = interp.get_attributes(), conv.get_attributes()
        scales = np.array([o / i for o, i in zip(list(interp.get_output_shape(0))[2:], list(frames.get_output_shape(0))[2:])], np.float32)
        up = ops.interpolate(
            x5,
            ops.constant(scales),
            "nearest",
            "scales",
            coordinate_transformation_mode=attrs["coordinate_transformation_mode"],
            nearest_mode=attrs["nearest_mode"],
            axes=_const_i64([3, 4]),
        )
        conv5 = ops.convolution(
            up,
            ops.constant(weight[:, :, None]),
            [1] + conv_attrs["strides"],
            [0] + conv_attrs["pads_begin"],
            [0] + conv_attrs["pads_end"],
            [1] + conv_attrs["dilations"],
            conv_attrs["auto_pad"],
        )
        conv5.set_friendly_name(conv.get_friendly_name() + "/5d")
        out = ops.add(conv5, ops.constant(_const(bias.input_value(1)).reshape(1, -1, 1, 1, 1)))
        if list(out.get_output_shape(0)) != list(to_5d.get_output_shape(0)):
            continue
        for target in list(to_5d.output(0).get_target_inputs()):
            target.replace_source_output(out.output(0))
        count += 1
    return count


def dupup_shortcut(model):
    """DupUp3D: channel repeat_interleave + 8D shuffle at full size  =>  small channel->depth shuffle + nearest upsampling."""
    count = 0
    for gather in model.get_ordered_ops():
        if gather.get_type_name() != "Gather" or "repeat_interleave" not in gather.get_friendly_name():
            continue
        x, axis = gather.input_value(0), _const(gather.input_value(2))
        if axis is None or int(axis) != 1 or len(x.get_shape()) != 5:
            continue
        _, c_in, t, h, w = x.get_shape()
        if gather.get_output_shape(0)[1] % c_in:
            continue
        repeats = gather.get_output_shape(0)[1] // c_in
        split = _single_consumer(gather, "Reshape")
        if split is None or len(split.get_output_shape(0)) != 8:
            continue
        perm = _single_consumer(split, "Transpose")
        if perm is None or _perm(perm) != [0, 1, 5, 2, 6, 3, 7, 4]:
            continue
        merges = [t.get_node() for t in perm.output(0).get_target_inputs() if t.get_node().get_type_name() == "Reshape"]
        _, c_out, *block = split.get_output_shape(0)[:5]
        if len(merges) != 1 or list(split.get_output_shape(0)[5:]) != [t, h, w]:
            continue
        # Channels c*R+r map to (o, a, b, c) blocks; R covers a suffix of the block dims.
        lead, dup, rest = list(block), [1, 1, 1], repeats
        for i in (2, 1, 0):
            if rest % block[i]:
                break
            rest //= block[i]
            lead[i], dup[i] = 1, block[i]
        if rest != 1 or c_out * int(np.prod(lead)) != c_in:
            continue
        y = x
        if any(v > 1 for v in lead):
            y = ops.reshape(y, _const_i64([1, c_out, *lead, t, h, w]), False).output(0)
            y = ops.transpose(y, _const_i64([0, 1, 5, 2, 6, 3, 7, 4])).output(0)
            y = ops.reshape(y, _const_i64([1, c_out, t * lead[0], h * lead[1], w * lead[2]]), False).output(0)
        axes = [i + 2 for i in range(3) if dup[i] > 1]
        if axes:
            y = ops.interpolate(
                y,
                ops.constant(np.array([dup[a - 2] for a in axes], np.float32)),
                "nearest",
                "scales",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
                axes=_const_i64(axes),
            ).output(0)
        if y == x or list(y.get_shape()) != list(merges[0].get_output_shape(0)):
            continue
        for target in list(merges[0].output(0).get_target_inputs()):
            target.replace_source_output(y)
        count += 1
    return count


def fold_zero_pads(model):
    """Fold constant zero Pads into the padding of the following Convolution."""
    from openvino.passes import ConstantFolding, Manager

    manager = Manager()
    manager.register_pass(ConstantFolding())
    manager.run_passes(model)
    count = 0
    for conv in model.get_ordered_ops():
        pad = conv.input_value(0).get_node() if conv.get_type_name() == "Convolution" else None
        if pad is None or pad.get_type_name() != "Pad" or pad.get_attributes().get("pad_mode") != "constant":
            continue
        if len(pad.output(0).get_target_inputs()) != 1 or conv.get_attributes()["auto_pad"] != "explicit":
            continue
        begin, end = _const(pad.input_value(1)), _const(pad.input_value(2))
        value = _const(pad.input_value(3)) if pad.get_input_size() > 3 else np.zeros(1)
        if begin is None or end is None or value is None or np.any(value):
            continue
        if np.any(begin[:2]) or np.any(end[:2]) or np.any(begin < 0) or np.any(end < 0):
            continue
        attrs = conv.get_attributes()
        new = ops.convolution(
            pad.input_value(0),
            conv.input_value(1),
            attrs["strides"],
            [int(a + b) for a, b in zip(attrs["pads_begin"], begin[2:])],
            [int(a + b) for a, b in zip(attrs["pads_end"], end[2:])],
            attrs["dilations"],
            "explicit",
        )
        new.set_friendly_name(conv.get_friendly_name())
        for target in list(conv.output(0).get_target_inputs()):
            target.replace_source_output(new.output(0))
        count += 1
    return count


def optimize_vae_decoder(model: ov.Model) -> dict[str, int]:
    """Apply all decoder graph rewrites; the model must be an f16 graph exported from a half-precision VAE."""
    counts = {
        "RMSNorm": rmsnorm_f16(model),
        "5D upsampler": upsampler_5d(model),
        "DupUp3D": dupup_shortcut(model),
        "zero Pad": fold_zero_pads(model),
    }
    model.validate_nodes_and_infer_types()
    return counts


# Convs kept in f16 by the per-layer INT8 sensitivity sweep (PSNR vs FP32 reference).
INT8_KEEP_F16 = (
    r"post_quant_conv|decoder\.conv_(in|out)|upsampler|up_blocks\.0\.resnets\.[12]\.conv2"
    r"|up_blocks\.1\.resnets\.[0-2]\.conv2|up_blocks\.2\.resnets\.1\.conv2|up_blocks\.3\.resnets\.[12]\.conv2"
)


def _conv_key(conv):
    return re.sub(r"/aten::_convolution/Convolution(_\d+)?$", "", conv.get_friendly_name())


def _fake_quantize(value, lo, hi, levels):
    dtype = value.get_element_type().to_dtype()
    lo, hi = ops.constant(np.asarray(lo, dtype=dtype)), ops.constant(np.asarray(hi, dtype=dtype))
    return ops.fake_quantize(value, lo, hi, lo, hi, levels)


def quantize_vae_decoder(model_path: str | Path, latent: np.ndarray, device: str = "GPU", keep_f16: str = INT8_KEEP_F16) -> tuple[ov.Model, int]:
    """Mixed INT8: symmetric per-tensor activations, per-channel weights; calibrated from one decoder run."""
    core = ov.Core()
    model = core.read_model(model_path)
    convs = [o for o in model.get_ordered_ops() if o.get_type_name() == "Convolution"]
    probes = []
    for conv in convs:
        act = conv.input_value(0)
        axes = _const_i64(np.arange(len(act.get_shape())))
        probes += [ops.reduce_min(act, axes, False).output(0), ops.reduce_max(act, axes, False).output(0)]
    model.add_outputs(probes)
    compiled = core.compile_model(model, device, {"INFERENCE_PRECISION_HINT": "f16"})
    result = compiled([latent.astype(np.float16)])
    values = [float(result[compiled.outputs[i + 1]]) for i in range(len(probes))]
    del compiled, model
    ranges = {}
    for i, conv in enumerate(convs):
        lo, hi = ranges.get(_conv_key(conv), (np.inf, -np.inf))
        ranges[_conv_key(conv)] = (min(lo, values[2 * i]), max(hi, values[2 * i + 1]))

    model = core.read_model(model_path)
    quantized = 0
    for conv in [o for o in model.get_ordered_ops() if o.get_type_name() == "Convolution"]:
        key = _conv_key(conv)
        weight_node = _skip_converts(conv.input_value(1)).get_node()
        if re.search(keep_f16, key) or weight_node.get_type_name() != "Constant":
            continue
        # Symmetric i8 (zero point 0): oneDNN has no int8 Conv3D with activation zero points.
        amax = max(abs(v) for v in ranges[key])
        weight = np.array(weight_node.get_data()).astype(np.float32)
        wmax = np.maximum(np.abs(weight).reshape(weight.shape[0], -1).max(axis=1), 1e-8).reshape([-1] + [1] * (weight.ndim - 1))
        conv.input(0).replace_source_output(_fake_quantize(conv.input_value(0), -amax, amax * 127 / 128, 256).output(0))
        conv.input(1).replace_source_output(_fake_quantize(conv.input_value(1), -wmax, wmax, 255).output(0))
        quantized += 1
    model.validate_nodes_and_infer_types()
    return model, quantized


def to_frames(video: np.ndarray) -> np.ndarray:
    """[1, 3, T, H, W] in [-1, 1] -> [T, H, W, 3] uint8."""
    return np.clip((video[0].transpose(1, 2, 3, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)


def save_video(video: np.ndarray, path: str | Path, fps: float = FPS) -> Path:
    """Write a [1, 3, T, H, W] video in [-1, 1] as an H.264 MP4."""
    import imageio.v2 as imageio

    imageio.mimwrite(path, list(to_frames(video)), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    return Path(path)
