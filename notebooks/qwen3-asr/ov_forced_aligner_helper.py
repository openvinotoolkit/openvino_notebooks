"""Conversion and optimization helpers for Qwen3-ForcedAligner with OpenVINO.

`convert_forced_aligner` exports the model with Optimum Intel, optionally compresses the weights
with NNCF, and applies decoder graph rewrites that do not change the model outputs:

* Causal SDPA: the exported decoder builds a [B, 1, L, L] causal mask in-graph and passes it to every
  ScaledDotProductAttention. The decoder has no padding input, so the mask is purely causal; it is
  replaced with SDPA(causal=True), which avoids materializing the mask on long audio.
* RoPE lookup table: in-graph Sin/Cos of position * inv_freq are replaced with cos/sin tables built
  in float64 and gathered by position.
* Q/K/V projection fusion (FP16 only): the three projections of each layer become one MatMul + split.

`enable_batched_encoder` makes the aligner run all 8 s encoder windows of a clip in one inference
request instead of one request per window. The windows are independent, so outputs are unchanged.
"""

import gc
import json
import re
import shutil
import tempfile
import types
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import openvino as ov
import torch
from openvino import opset13 as ops

if TYPE_CHECKING:
    from optimum.intel import OVModelForQwen3ASRForcedAligner

WEIGHT_FORMATS = ("FP16", "INT8", "INT4")
CONFIG_FILES = ("config.json", "generation_config.json")
PROJECTION_PATTERN = re.compile(r"layers\.(\d+)\.self_attn\.([qkv])_proj/")


def use_causal_sdpa(model: ov.Model) -> int:
    """
    Replace the explicit causal attention mask of every ScaledDotProductAttention with ``causal=True``.

    :param model: decoder model, modified in place
    :returns: number of replaced attention layers
    """
    replaced = 0
    for sdpa in [op for op in model.get_ordered_ops() if op.get_type_name() == "ScaledDotProductAttention"]:
        if sdpa.get_attributes().get("causal", False) or len(sdpa.inputs()) != 5:
            continue
        q, k, v = (sdpa.input_value(i) for i in range(3))
        zero_mask = ops.constant(np.array(0, dtype=np.float32))
        if q.get_element_type() != zero_mask.get_output_element_type(0):
            zero_mask = ops.convert(zero_mask, q.get_element_type())
        new = ops.scaled_dot_product_attention(q, k, v, zero_mask, sdpa.input_value(4), causal=True, name=sdpa.get_friendly_name())
        sdpa.output(0).replace(new.output(0))
        replaced += 1
    model.validate_nodes_and_infer_types()
    return replaced


def _rope_source(trig: ov.Node) -> Tuple[np.ndarray, ov.Output]:
    """
    Find the RoPE inverse frequencies and position source for angles = Concat(Transpose(inv_freq @ positions)) x 2.

    :param trig: Cos or Sin node of the rotary embedding
    :returns: inverse frequencies as float64 and the output producing the positions
    """
    concat = trig.input_value(0).get_node()
    if concat.get_type_name() != "Concat":
        raise ValueError(f"Unexpected RoPE input {concat.get_type_name()}")
    transpose = concat.input_value(0).get_node()
    matmul = transpose.input_value(0).get_node()
    if transpose.get_type_name() != "Transpose" or matmul.get_type_name() != "MatMul":
        raise ValueError("Unexpected RoPE angle subgraph")
    inv_freq = matmul.input_value(0).get_node()
    if inv_freq.get_type_name() != "Constant":
        raise ValueError("inv_freq is not a constant")
    node = matmul.input_value(1).get_node()
    while node.get_type_name() in ("Convert", "Unsqueeze"):
        node = node.input_value(0).get_node()
    if node.get_type_name() != "Range":
        raise ValueError("RoPE positions are not a Range (position_ids must be 0..L-1)")
    return inv_freq.get_data().reshape(-1).astype(np.float64), node.output(0)


def use_rope_table(model: ov.Model, max_positions: int) -> int:
    """
    Replace the in-graph RoPE Cos/Sin computation with precomputed cos/sin tables gathered by position.

    :param model: decoder model, modified in place
    :param max_positions: number of positions in the tables
    :returns: number of positions in the tables
    """
    trig = {op.get_type_name(): op for op in model.get_ops() if op.get_type_name() in ("Cos", "Sin")}
    if set(trig) != {"Cos", "Sin"}:
        raise ValueError("Expected one Cos and one Sin RoPE node")
    inv_freq, positions = _rope_source(trig=trig["Cos"])
    angles = np.arange(max_positions, dtype=np.float64)[:, None] * inv_freq[None, :]
    angles = np.concatenate([angles, angles], axis=-1)
    for name, fn in (("Cos", np.cos), ("Sin", np.sin)):
        table = ops.constant(fn(angles).astype(np.float32), name=f"rope_{name.lower()}_table")
        gathered = ops.gather(table, positions, ops.constant(np.array(0, dtype=np.int64)))
        out = ops.unsqueeze(gathered, ops.constant(np.array([0], dtype=np.int64)), name=f"rope_{name.lower()}_lookup")
        trig[name].output(0).replace(out.output(0))
    model.validate_nodes_and_infer_types()
    return max_positions


def fuse_qkv_projections(model: ov.Model) -> int:
    """
    Fuse the Q, K and V projection MatMuls of each decoder layer into one MatMul followed by a split.

    :param model: decoder model with uncompressed projection weights, modified in place
    :returns: number of fused layers
    """
    groups = defaultdict(dict)
    for node in model.get_ops():
        if node.get_type_name() == "MatMul":
            match = PROJECTION_PATTERN.search(node.get_friendly_name())
            if match:
                groups[int(match.group(1))][match.group(2)] = node
    if not groups:
        raise ValueError("No decoder self-attention Q/K/V projections found")

    for layer, projections in groups.items():
        if set(projections) != {"q", "k", "v"}:
            raise ValueError(f"Layer {layer} does not contain exactly one Q, K, and V projection")
        q, k, v = (projections[name] for name in ("q", "k", "v"))
        if not (q.input_value(0).get_node() is k.input_value(0).get_node() is v.input_value(0).get_node()):
            raise ValueError(f"Layer {layer} Q/K/V projections do not share an input")
        if any(node.get_attributes() != q.get_attributes() for node in (k, v)) or q.get_attributes() != {
            "transpose_a": False,
            "transpose_b": True,
        }:
            raise ValueError(f"Layer {layer} has an unsupported Q/K/V MatMul layout")
        widths = []
        for node in (q, k, v):
            shape = node.input_value(1).get_partial_shape()
            if shape.rank.get_length() != 2 or not shape.is_static:
                raise ValueError(f"Layer {layer} has a non-static projection weight")
            widths.append(shape[0].get_length())

        weights = ops.concat([q.input_value(1), k.input_value(1), v.input_value(1)], axis=0, name=f"qkv_fused_layer_{layer}_weights")
        fused = ops.matmul(q.input_value(0), weights, transpose_a=False, transpose_b=True, name=f"qkv_fused_layer_{layer}_matmul")
        split = ops.variadic_split(
            fused,
            ops.constant(np.array(-1, dtype=np.int64)),
            ops.constant(np.asarray(widths, dtype=np.int64)),
            name=f"qkv_fused_layer_{layer}_split",
        )
        for index, node in enumerate((q, k, v)):
            node.output(0).replace(split.output(index))
    model.validate_nodes_and_infer_types()
    return len(groups)


def _compress(model: ov.Model, weight_format: str) -> ov.Model:
    """
    Compress model weights with NNCF.

    :param model: model to compress
    :param weight_format: "INT8" (asymmetric, per-channel) or "INT4" (symmetric, group size 64)
    :returns: compressed model
    """
    import nncf

    if weight_format == "INT8":
        return nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    return nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT4_SYM, ratio=1.0, group_size=64)


def convert_forced_aligner(model_id: str, output_dir: Path, weight_format: str = "INT8") -> Path:
    """
    Export Qwen3-ForcedAligner to OpenVINO IR, optionally compress its weights with NNCF, and optimize the decoder graph.

    :param model_id: Hugging Face model id or local path of the model
    :param output_dir: directory for the optimized model; conversion is skipped if it already contains one
    :param weight_format: "FP16", "INT8" or "INT4"
    :returns: path to the optimized model directory
    """
    output_dir = Path(output_dir)
    if weight_format not in WEIGHT_FORMATS:
        raise ValueError(f"weight_format must be one of {WEIGHT_FORMATS}")
    if (output_dir / "openvino_decoder_model.xml").exists():
        print(f"Optimized model already exists in {output_dir}")
        return output_dir

    from optimum.intel import OVModelForQwen3ASRForcedAligner

    core = ov.Core()
    with tempfile.TemporaryDirectory(dir=output_dir.parent if output_dir.parent.exists() else None) as tmp:
        export_dir = Path(tmp)
        print(f"Exporting {model_id} to OpenVINO IR...")
        model = OVModelForQwen3ASRForcedAligner.from_pretrained(model_id, export=True, load_in_8bit=False, compile=False)
        model.save_pretrained(export_dir)
        del model
        gc.collect()

        config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
        max_positions = int(config.get("text_config", config)["max_position_embeddings"])

        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in CONFIG_FILES:
            if (export_dir / filename).exists():
                shutil.copy2(export_dir / filename, output_dir / filename)

        encoder = core.read_model(export_dir / "openvino_encoder_model.xml")
        if weight_format != "FP16":
            print(f"Compressing encoder weights to {weight_format} with NNCF...")
            encoder = _compress(model=encoder, weight_format=weight_format)
        ov.save_model(encoder, output_dir / "openvino_encoder_model.xml", compress_to_fp16=False)
        del encoder

        decoder = core.read_model(export_dir / "openvino_decoder_model.xml")
        if weight_format != "FP16":
            # Compress before the rewrites so the float RoPE cos/sin tables are not quantized.
            print(f"Compressing decoder weights to {weight_format} with NNCF...")
            decoder = _compress(model=decoder, weight_format=weight_format)
        summary = {
            "causal_sdpa_layers": use_causal_sdpa(model=decoder),
            "rope_table_positions": use_rope_table(model=decoder, max_positions=max_positions),
        }
        if weight_format == "FP16":
            # Compressed INT8/INT4 projections are already fused by the GPU plugin; FP16 ones are not.
            summary["qkv_fused_layers"] = fuse_qkv_projections(model=decoder)
        ov.save_model(decoder, output_dir / "openvino_decoder_model.xml", compress_to_fp16=False)
        del decoder
        gc.collect()
    print(f"Decoder graph optimizations: {summary}")
    print(f"Optimized {weight_format} model saved to {output_dir}")
    return output_dir


def _batched_chunked_forward(self, input_features: torch.Tensor, feature_mask: torch.Tensor | None, n_window: int, n_window_infer: int) -> torch.Tensor:
    """
    Batched variant of ``OVEncoder.chunked_forward`` from optimum-intel (Apache License 2.0,
    https://github.com/huggingface/optimum-intel): all encoder windows run in one inference request.

    :param input_features: mel features of one clip, [n_mels, frames] or [1, n_mels, frames]
    :param feature_mask: valid-frame mask, or None if all frames are valid
    :param n_window: half of the chunk length in frames
    :param n_window_infer: encoder window length in frames
    :returns: encoder hidden states of the valid frames, [1, tokens, hidden_size]
    """
    import torch.nn.functional as F

    chunk_len = n_window * 2
    chunks_per_window = n_window_infer // chunk_len
    post_cnn_per_chunk = self._post_cnn_length(chunk_len)
    if input_features.dim() == 2:
        input_features = input_features.unsqueeze(0)
    feature_lens = input_features.shape[-1]
    num_chunks_total = (feature_lens + chunk_len - 1) // chunk_len
    mask = feature_mask if feature_mask is not None else torch.ones(feature_lens, dtype=torch.long)
    mask = mask.reshape(-1)[:feature_lens].to(torch.long)
    mask = F.pad(mask, (0, num_chunks_total * chunk_len - feature_lens))
    chunk_valid_lens = mask.view(num_chunks_total, chunk_len).sum(dim=-1)

    num_windows = (feature_lens + n_window_infer - 1) // n_window_infer
    padded = F.pad(input_features[0], (0, num_windows * n_window_infer - feature_lens))
    windows = padded.reshape(padded.shape[0], num_windows, n_window_infer).permute(1, 0, 2)
    hidden = self.request({"input_features": np.ascontiguousarray(windows.numpy(), dtype=np.float32)})
    hidden = torch.from_numpy(hidden["last_hidden_state"]).to(self.device)
    hidden = hidden.reshape(num_windows * chunks_per_window, post_cnn_per_chunk, -1)

    features = []
    for chunk in range(num_chunks_total):
        keep = int(self._post_cnn_length(int(chunk_valid_lens[chunk].item())))
        if keep > 0:
            features.append(hidden[chunk, :keep])
    return torch.cat(features, dim=0).view(1, -1, features[0].shape[-1])


def enable_batched_encoder(aligner_model: "OVModelForQwen3ASRForcedAligner") -> "OVModelForQwen3ASRForcedAligner":
    """
    Run all encoder windows of a clip in one inference request instead of one request per window.

    Only this model instance is changed; other models (for example the ASR model) keep the default behavior.

    :param aligner_model: loaded ``OVModelForQwen3ASRForcedAligner``
    :returns: the same model, for chaining
    """
    aligner_model.encoder.chunked_forward = types.MethodType(_batched_chunked_forward, aligner_model.encoder)
    return aligner_model
