"""OpenVINO conversion and inference helpers for baidu/Unlimited-OCR.

Unlimited-OCR is a vision-language OCR model composed of:
  * a SAM ViT-B + CLIP-L vision encoder stack and a linear projector (2048 -> 1280),
  * a DeepSeek-V2 *MoE* language model (12 layers, 64 routed + 2 shared experts, top-6)
    that, because ``use_mla=False``, uses ``SlidingWindowLlamaAttention`` on every layer.

This module mirrors the structure of ``ov_deepseek_ocr_helper.py`` but adapts three
model-specific points:
  1. The language model is loaded with ``attn_implementation="eager"`` so the decoder
     layers pick the standard (Llama-style q/k/v/o) attention, then each attention is
     patched with a static, trace-friendly forward (no ``.item()``, no ring-buffer state).
  2. The sliding window (size 128) is reproduced faithfully at inference time with an
     additive 4D attention mask built in :meth:`OvModelForCausalLMWithEmb.prepare_inputs`:
     every query attends to *all* prefill tokens plus the last 128 generated tokens.
  3. The MoE expert routing is replaced with a vectorised, statically-traceable loop over
     all experts (one-hot mask + ``index_add_``) honouring ``routed_scaling_factor`` and
     ``norm_topk_prob`` so the numerics match the original gate.
"""

import gc
import math
import os
import re
import sys
import types
from abc import ABC
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import openvino as ov
import openvino.opset13 as opset13
import torch
import torch.nn as nn
import nncf
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm
from transformers import GenerationConfig, GenerationMixin, PretrainedConfig, TextStreamer
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from torchvision import transforms

# The original modeling code is downloaded from the HF Hub into this local folder by the
# notebook (snapshot_download of "baidu/Unlimited-OCR").  We import a couple of utilities
# (conversation template + config class) from it, exactly like the DeepSeek-OCR notebook.
MODEL_DIR_NAME = "Unlimited_OCR"

model_ids = ["baidu/Unlimited-OCR"]
model_id = model_ids[0]
model_path = Path(model_id.split("/")[-1])

VISION_EMBEDDINGS_PATH = "openvino_vision_embeddings_model.xml"
VISION_CROP_EMBEDDINGS_PATH = "openvino_vision_crop_embeddings_model.xml"
TEXT_EMBEDDINGS_PATH = "openvino_text_embeddings_model.xml"
LANGUAGE_MODEL_PATH = "openvino_language_model.xml"

# The vision encoder is exported at two fixed square resolutions because the SAM/CLIP
# positional-embedding interpolation is resolution dependent and cannot be traced as a
# single dynamic-resolution graph. These cover the model's default "gundam" pipeline:
#   * GLOBAL view  -> base_size  (1024x1024)
#   * CROP tiles   -> image_size (640x640), variable tile count via dynamic batch
GLOBAL_VIEW_SIZE = 1024
CROP_VIEW_SIZE = 640

IMAGE_TOKEN_ID = 128815

core = ov.Core()


# --------------------------------------------------------------------------------------
# Image / text utilities (shared with the original pipeline)
# --------------------------------------------------------------------------------------
def load_image(image_path):
    try:
        image = Image.open(image_path)
        return ImageOps.exif_transpose(image)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except Exception:  # noqa: BLE001
            return None


def re_match(text):
    ref_pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(ref_pattern, text, re.DOTALL)

    det_pattern = r"(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)"
    for full_match, label, box in re.findall(det_pattern, text, re.DOTALL):
        matches.append((full_match, label, box))

    mathes_image, mathes_other = [], []
    for a_match in matches:
        if a_match[1].strip() == "image" or "<|ref|>image<|/ref|>" in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])  # noqa: S307 - trusted model output
        if cor_list and isinstance(cor_list[0], (int, float)):
            cor_list = [cor_list]
    except Exception as e:  # noqa: BLE001
        print(e)
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, ouput_path, image_prefix=""):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    img_idx = 0

    for ref in refs:
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if not result:
                continue
            label_type, points_list = result
            color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
            color_a = color + (20,)
            for points in points_list:
                x1, y1, x2, y2 = points
                x1 = int(x1 / 999 * image_width)
                y1 = int(y1 / 999 * image_height)
                x2 = int(x2 / 999 * image_width)
                y2 = int(y2 / 999 * image_height)
                if label_type == "image":
                    try:
                        image.crop((x1, y1, x2, y2)).save(f"{ouput_path}/images/{image_prefix}{img_idx}.jpg")
                    except Exception as e:  # noqa: BLE001
                        print(e)
                    img_idx += 1
                try:
                    width = 4 if label_type == "title" else 2
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                    draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                    text_x, text_y = x1, max(0, y1 - 15)
                    text_bbox = draw.textbbox((0, 0), label_type, font=font)
                    draw.rectangle(
                        [text_x, text_y, text_x + (text_bbox[2] - text_bbox[0]), text_y + (text_bbox[3] - text_bbox[1])],
                        fill=(255, 255, 255, 30),
                    )
                    draw.text((text_x, text_y), label_type, font=font, fill=color)
                except Exception:  # noqa: BLE001, S110
                    pass
        except Exception:  # noqa: BLE001, S112
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path, image_prefix=""):
    return draw_bounding_boxes(image, ref_texts, output_path, image_prefix=image_prefix)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=32, image_size=640, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images, target_aspect_ratio


def normalize_transform(mean, std):
    if mean is None and std is None:
        return None
    if mean is None:
        mean = [0.0] * len(std)
    if std is None:
        std = [1.0] * len(mean)
    return transforms.Normalize(mean=mean, std=std)


def format_messages(conversations, sft_format="deepseek", system_prompt=""):
    from Unlimited_OCR.conversation import get_conv_template

    conv = get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    return conv.get_prompt().strip()


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    if bos:
        t = [0] + t
    if eos:
        t = t + [1]
    return t


def load_pil_images(conversations):
    pil_images = []
    for message in conversations:
        if "images" not in message:
            continue
        for image_path in message["images"]:
            pil_img = load_image(image_path).convert("RGB")
            pil_images.append(pil_img)
    return pil_images


class BaseTransform(ABC):
    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


class BasicImageTransform(BaseTransform):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True):
        self.mean = mean
        self.std = std
        pipelines = [transforms.ToTensor()]
        norm = normalize_transform(mean, std) if normalize else nn.Identity()
        if norm is not None:
            pipelines.append(norm)
        self.transform = transforms.Compose(pipelines)

    def __call__(self, x):
        return self.transform(x)


class NoEOSTextStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):
        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")


class SlidingWindowNoRepeatNgramProcessor:
    """Block n-gram repetitions within a sliding window.

    Verbatim port of the original model's ``SlidingWindowNoRepeatNgramProcessor`` (aligned
    with SGLang's ``DeepseekOCRNoRepeatNGramLogitProcessor``). The original ``infer`` uses
    this instead of HuggingFace's plain ``no_repeat_ngram_size`` so we reproduce it exactly.
    """

    def __init__(self, ngram_size, window, whitelist_token_ids=None):
        self.ngram_size = ngram_size
        self.window = window
        self.whitelist = set(whitelist_token_ids) if whitelist_token_ids else set()

    def __call__(self, input_ids, scores):
        for batch_idx in range(input_ids.shape[0]):
            sequence = input_ids[batch_idx].tolist()
            if len(sequence) < self.ngram_size:
                continue
            search_start = max(0, len(sequence) - self.window)
            search_end = len(sequence) - self.ngram_size + 1
            if search_end <= search_start:
                continue
            current_prefix = tuple(sequence[-(self.ngram_size - 1):]) if self.ngram_size > 1 else tuple()
            banned = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx:idx + self.ngram_size]
                if self.ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                    banned.add(ngram[-1])
            banned.difference_update(self.whitelist)
            for token_id in banned:
                scores[batch_idx, token_id] = float("-inf")
        return scores


# --------------------------------------------------------------------------------------
# Stateful-model helpers (verbatim from the OpenVINO stateful-LLM recipe)
# --------------------------------------------------------------------------------------
def model_has_input_output_name(ov_model: ov.Model, name: str):
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, gather_dim):
    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(ov_model, not_kv_inputs, key_value_input_names, key_value_output_names, batch_dim, num_attention_heads, num_beams_and_batch=None):
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}
    if num_beams_and_batch is not None:
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)
    if num_beams_and_batch is not None:
        ov_model.validate_nodes_and_infer_types()
    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[2:-1]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[1:]]
    not_kv_inputs = [inp for inp in ov_model.inputs if not any(name in key_value_input_names for name in inp.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, 0)
    make_stateful(ov_model, not_kv_inputs, key_value_input_names, key_value_output_names, 0, 1, None)


def cleanup_torchscript_cache():
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


# --------------------------------------------------------------------------------------
# Trace-friendly patches for the DeepSeek-V2 MoE language model
# --------------------------------------------------------------------------------------
def llama_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """Static replacement for ``SlidingWindowLlamaAttention.forward``.

    Standard Llama attention: q/k/v/o projections, RoPE, a growing KV cache via
    ``past_key_value.update`` and the additive 4D attention mask that the (stock)
    base-model forward builds from the 2D mask supplied by the pipeline.  The sliding
    window is encoded in that mask, so there is no data-dependent control flow here
    and the graph stays static.
    """
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb as _apply_rotary,
        repeat_kv as _repeat_kv,
    )

    bsz, q_len, _ = hidden_states.size()
    num_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    num_kv_groups = self.num_key_value_groups

    query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = _apply_rotary(query_states, key_states, cos, sin)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

    key_states = _repeat_kv(key_states, num_kv_groups)
    value_states = _repeat_kv(value_states, num_kv_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def moe_forward(self, hidden_states):
    """Static MoE: loop over every expert with a one-hot mask + ``index_add_``.

    Mirrors the original ``MoEGate`` numerics (softmax scoring, greedy top-k,
    optional ``norm_topk_prob`` and ``routed_scaling_factor``) plus the shared experts.
    """
    identity = hidden_states
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = torch.nn.functional.linear(hidden_states.float(), self.gate.weight.float(), None)
    scores = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(scores, self.gate.top_k, dim=-1, sorted=False)
    if self.gate.top_k > 1 and self.gate.norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
    routing_weights = routing_weights * self.gate.routed_scaling_factor
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=len(self.experts)).permute(2, 1, 0)
    for expert_idx in range(len(self.experts)):
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    if self.config.n_shared_experts is not None:
        final_hidden_states = final_hidden_states + self.shared_experts(identity)
    return final_hidden_states


def deepseek_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    images=None,
    images_seq_mask=None,
    images_spatial_crop=None,
    return_dict=None,
):
    """Static decoder loop that forwards the *given* 4D additive mask to each layer.

    The original ``DeepseekV2Model.forward`` rebuilds (and at decode drops) the causal
    mask; here we keep the externally-supplied mask so the sliding window can be encoded
    in it.  KV cache is routed through ``DynamicCache`` and returned as a *legacy* tuple
    of tensors so the TorchScript tracer (which sees ``return_dict=False`` under
    ``torchscript=True``) can flatten the KV outputs.
    """
    cache = DynamicCache.from_legacy_cache(past_key_values)
    hidden_states = inputs_embeds
    for decoder_layer in self.layers:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
        )[0]
    hidden_states = self.norm(hidden_states)
    return (hidden_states, cache.to_legacy_cache())


# --------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------
def convert_unlimited_ocr(model_id=model_id, model_path=None, quantization_config=None):
    from transformers import AutoModel, AutoTokenizer

    if model_path is None:
        model_path = Path(model_id.split("/")[-1])
    model_path = Path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(model_path)

    if all((model_path / name).exists() for name in [VISION_EMBEDDINGS_PATH, VISION_CROP_EMBEDDINGS_PATH, TEXT_EMBEDDINGS_PATH, LANGUAGE_MODEL_PATH]):
        print(f"✅ {model_id} model already converted. You can find results in {model_path}")
        return model_path

    print(f"⌛ {model_id} conversion started. Be patient, it may take some time.")
    print("⌛ Load Original model")
    # eager attn_implementation -> decoder layers select SlidingWindowLlamaAttention (mha_eager)
    pt_model = AutoModel.from_pretrained(
        model_id, device_map="cpu", trust_remote_code=True, use_safetensors=True, attn_implementation="eager"
    )
    pt_model = pt_model.eval().to(torch.float32)
    config = pt_model.config
    config.image_newline = pt_model.model.image_newline.tolist()
    config.view_seperator = pt_model.model.view_seperator.tolist()
    # window used by the pipeline to build the sliding-window mask
    config.ring_window = getattr(config, "sliding_window_size", None) or getattr(config, "sliding_window", None)
    config.save_pretrained(model_path)
    __make_16bit_traceable(pt_model)
    print("✅ Original model successfully loaded")

    if not (model_path / TEXT_EMBEDDINGS_PATH).exists():
        print("⌛ Convert Input embedding model")
        ov_model = ov.convert_model(pt_model.model.get_input_embeddings(), example_input=torch.ones([2, 2], dtype=torch.long))
        ov.save_model(ov_model, model_path / TEXT_EMBEDDINGS_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Input embedding model successfully converted")

    if not all((model_path / name).exists() for name in [VISION_EMBEDDINGS_PATH, VISION_CROP_EMBEDDINGS_PATH]):

        def vision_forward(self, patches):
            features_1 = self.sam_model(patches)
            features_2 = self.vision_model(patches, features_1)
            features = torch.cat((features_2[:, 1:], features_1.flatten(2).permute(0, 2, 1)), dim=-1)
            return self.projector(features)

        # The SAM/CLIP positional-embedding helpers interpolate based on the input resolution
        # (``if src_size != tgt_size``). Tracing freezes that branch, so one IR cannot serve
        # both resolutions. We export two fixed-square-resolution IRs instead — one for the
        # 1024 global view and one for the 640 crop tiles — each with a dynamic batch
        # dimension so the variable number of crop tiles is handled at runtime. This mirrors
        # the multi-resolution vision split used by the DeepSeek-OCR-2 notebook.
        pt_model.model._orig_forward = pt_model.model.forward
        pt_model.model.forward = types.MethodType(vision_forward, pt_model.model)
        try:
            for size, path in [(GLOBAL_VIEW_SIZE, VISION_EMBEDDINGS_PATH), (CROP_VIEW_SIZE, VISION_CROP_EMBEDDINGS_PATH)]:
                if (model_path / path).exists():
                    continue
                print(f"⌛ Convert Image embedding model ({size}x{size})")
                # batch>1 example so the SAM positional-embedding broadcast (pos_embed batch 1
                # vs. input batch=num_tiles) is captured as a broadcast, then relax batch to -1.
                ov_model = ov.convert_model(
                    pt_model.model,
                    example_input=torch.ones([2, 3, size, size]),
                    input=[[-1, 3, size, size]],
                )
                if quantization_config is not None and "vision" in quantization_config:
                    ov_model = nncf.compress_weights(ov_model, **quantization_config["vision"])
                ov.save_model(ov_model, model_path / path)
                del ov_model
                cleanup_torchscript_cache()
                gc.collect()
                print(f"✅ Image embedding model ({size}x{size}) successfully converted")
        finally:
            pt_model.model.forward = pt_model.model._orig_forward
            del pt_model.model._orig_forward
        print("✅ Image embedding model successfully converted")

    if not (model_path / LANGUAGE_MODEL_PATH).exists():
        print("⌛ Convert Language model")
        lm = pt_model
        # patch attention + MoE on every decoder layer, and the base model's forward
        for block in lm.model.layers:
            block.self_attn.forward = types.MethodType(llama_attn_forward, block.self_attn)
            if hasattr(block.mlp, "moe_infer"):
                block.mlp.forward = types.MethodType(moe_forward, block.mlp)
        lm.model._orig_forward = lm.model.forward
        lm.model.forward = types.MethodType(deepseek_model_forward, lm.model)

        head_dim = lm.config.hidden_size // lm.config.num_attention_heads
        num_layers = lm.config.num_hidden_layers
        kv_shape = (2, lm.config.num_key_value_heads, 2, head_dim)

        inputs_embeds = torch.zeros([2, 2, lm.config.hidden_size], dtype=torch.float32)
        # explicit 4D additive mask: [batch, 1, q_len, kv_len(=past+cur)]
        attention_mask = torch.zeros([2, 1, 2, 4], dtype=torch.float32)
        position_ids = torch.tensor([[2, 3], [2, 3]])
        pkv_inputs, pkv_input_names, pkv_output_names = [], [], []
        for idx in range(num_layers):
            pkv_inputs.append((torch.randn(kv_shape), torch.randn(kv_shape)))
            pkv_input_names.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
            pkv_output_names.extend([f"present.{idx}.key", f"present.{idx}.value"])

        model_inputs = ["attention_mask", "position_ids", *pkv_input_names, "inputs_embeds"]
        model_outputs = ["logits", *pkv_output_names]

        lm.config.torchscript = True
        dummy_inputs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": pkv_inputs,
            "inputs_embeds": inputs_embeds,
        }
        ts_decoder = TorchScriptPythonDecoder(lm, example_input=dummy_inputs, trace_kwargs={"check_trace": False})
        ov_model = ov.convert_model(ts_decoder, example_input=dummy_inputs)
        for inp, name in zip(ov_model.inputs, model_inputs):
            inp.get_tensor().set_names({name})
        for out, name in zip(ov_model.outputs, model_outputs):
            out.get_tensor().set_names({name})
        patch_stateful(ov_model)
        print("✅ Language model successfully converted")

        if quantization_config is not None and "llm" in quantization_config:
            print(f"⌛ Weights compression with {quantization_config['llm']['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config["llm"])
            print("✅ Weights compression finished")
        else:
            ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
        ov.save_model(ov_model, model_path / LANGUAGE_MODEL_PATH)
        del ov_model
        lm.model.forward = lm.model._orig_forward
        del lm.model._orig_forward
        cleanup_torchscript_cache()
        gc.collect()

    del pt_model
    gc.collect()
    print(f"✅ {model_id} model conversion finished. You can find results in {model_path}")
    return model_path


# --------------------------------------------------------------------------------------
# Inference wrappers
# --------------------------------------------------------------------------------------
class OvModelForCausalLMWithEmb(GenerationMixin):
    """Stateful OpenVINO language model wrapper with sliding-window attention masking."""

    def __init__(self, model_dir, device="CPU", config=None, ov_config=None, compile=True):
        self._supports_cache_class = False
        self.config = config
        if isinstance(self.config, dict):
            self.config = PretrainedConfig.from_dict(self.config)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / LANGUAGE_MODEL_PATH)
        self.token_emb = core.read_model(model_dir / TEXT_EMBEDDINGS_PATH)
        self.request = None
        self.token_emb_request = None
        self._device = device.upper()
        self.device = torch.device("cpu")
        self.ov_config = ov_config or {"KV_CACHE_PRECISION": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}
        self.next_beam_idx = None
        self._past_length = None
        self._prefill_length = 0
        # sliding-window size (faithful: query sees all prefill + last W generated tokens)
        self._ring_window = getattr(self.config, "ring_window", None) or getattr(self.config, "sliding_window_size", None) or getattr(self.config, "sliding_window", None)
        self.input_names = [t.get_any_name() for t in self.model.inputs]
        self.main_input_name = "input_ids"
        if compile:
            self.compile()

    def compile(self):
        if self.request is None:
            self.request = core.compile_model(self.model, self._device, self.ov_config).create_infer_request()
        self._compile_token_emb()

    def _compile_token_emb(self):
        if self.token_emb_request is None:
            self.token_emb_request = core.compile_model(self.token_emb, self._device, self.ov_config)

    def to(self, device):
        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()
        return self

    def clear_requests(self):
        del self.request
        del self.token_emb_request
        self.request = None
        self.token_emb_request = None

    def embed_tokens(self, input_ids: torch.LongTensor):
        self._compile_token_emb()
        return self.token_emb_request(input_ids, share_inputs=True)[0]

    def _build_attention_mask(self, q_len, past_len):
        """Additive 4D mask [1, 1, q_len, kv_len] encoding causal + sliding window."""
        kv_len = past_len + q_len
        neg = np.finfo(np.float32).min
        W = self._ring_window
        if past_len == 0:
            # prefill: pure causal (window disabled, like the original ring buffer)
            mask = np.triu(np.full((q_len, kv_len), neg, dtype=np.float32), k=1)
            return mask[None, None]
        mask = np.zeros((1, 1, q_len, kv_len), dtype=np.float32)
        key_pos = np.arange(kv_len)
        for i in range(q_len):
            qpos = past_len + i
            allowed = key_pos <= qpos
            if W is not None:
                allowed &= (key_pos < self._prefill_length) | (key_pos >= (qpos + 1 - W))
            mask[0, 0, i, ~allowed] = neg
        return mask

    def prepare_inputs(self, input_ids, attention_mask=None, past_key_values=None, position_ids=None, inputs_embeds=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        inputs = {}
        if past_key_values is None:
            if self.request is not None:
                self.request.reset_state()
                self.next_beam_idx = np.arange(batch_size, dtype=int)
                self._past_length = 0
                self._prefill_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
        past_len = self._get_past_length(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids if past_key_values is None else input_ids[:, -1:])
            if hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb
        inputs["inputs_embeds"] = inputs_embeds

        q_len = inputs_embeds.shape[1]
        inputs["attention_mask"] = self._build_attention_mask(q_len, past_len)

        if "position_ids" in self.input_names:
            if position_ids is None:
                base_mask = np.ones((inputs_embeds.shape[0], q_len + past_len), dtype=int)
                position_ids = np.cumsum(base_mask, axis=1) - 1
                position_ids = position_ids[:, -q_len:]
            else:
                position_ids = np.array(position_ids)
            inputs["position_ids"] = position_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
        return inputs

    def forward(self, input_ids, attention_mask=None, past_key_values=None, position_ids=None, inputs_embeds=None, **kwargs):
        self.compile()
        inputs = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)
        self._past_length += inputs["inputs_embeds"].shape[1]
        return CausalLMOutputWithPast(logits=logits, past_key_values=((),))

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)
        past_len = 0
        if past_key_values is not None:
            past_len = self._get_past_length(past_key_values)
            if attention_mask is not None and input_ids is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_len):]
            elif input_ids is not None and past_len < input_ids.shape[1]:
                input_ids = input_ids[:, past_len:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None and "position_ids" in self.input_names:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values and input_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        cache_position = torch.arange(past_len, past_len + position_ids.shape[-1], device=position_ids.device)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds if past_key_values is None else None,
            "cache_position": cache_position,
        }

    def _get_past_length(self, past_key_values=None):
        return 0 if past_key_values is None else self._past_length

    def _reorder_cache(self, past_key_values, beam_idx):
        self.next_beam_idx = np.array(beam_idx)
        return past_key_values

    def can_generate(self):
        return True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class OVUnlimitedOCRForCausalLM(GenerationMixin):
    """High-level OpenVINO pipeline mirroring the original ``UnlimitedOCRForCausalLM.infer``."""

    def __init__(self, model_dir, device="CPU", ov_config=None):
        from Unlimited_OCR.modeling_unlimitedocr import UnlimitedOCRConfig

        model_dir = Path(model_dir)
        self.config = UnlimitedOCRConfig.from_pretrained(model_dir)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.language_model = OvModelForCausalLMWithEmb(model_dir, device, self.config, ov_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"
        self._supports_cache_class = False
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        _vis_cfg = {"INFERENCE_PRECISION_HINT": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}
        # global view (1024) and crop tiles (640) use separate fixed-resolution IRs
        self.vision_embeddings = core.compile_model(model_dir / VISION_EMBEDDINGS_PATH, device, _vis_cfg)
        self.vision_crop_embeddings = core.compile_model(model_dir / VISION_CROP_EMBEDDINGS_PATH, device, _vis_cfg)
        self.image_newline = torch.tensor(self.config.image_newline)
        self.view_seperator = torch.tensor(self.config.view_seperator)

    def _encode_views(self, views):
        # pick the matching fixed-resolution vision IR by spatial size (1024 global vs 640 crop)
        size = int(views.shape[-1])
        req = self.vision_embeddings if size >= GLOBAL_VIEW_SIZE else self.vision_crop_embeddings
        return torch.from_numpy(req(views)[0])

    def prepare_inputs_embeds(self, input_ids, images=None, images_seq_mask=None, images_spatial_crop=None, **ignore):
        if images is None or images_spatial_crop.sum() == 0:
            return torch.from_numpy(self.language_model.embed_tokens(input_ids))

        inputs_embeds = torch.from_numpy(self.language_model.embed_tokens(input_ids))
        idx = 0
        for image, crop_shape in zip(images, images_spatial_crop):
            images_in_this_batch = []
            patches = image[0]
            image_ori = image[1]
            with torch.no_grad():
                if torch.sum(patches).item() != 0:
                    # crop tiles -> 640 IR, global view -> 1024 IR (selected by spatial size)
                    local_features = self._encode_views(patches)
                    global_features = self._encode_views(image_ori)

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)
                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2**0.5)
                    width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat([global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1)
                    global_features = global_features.view(-1, n_dim)

                    local_features = (
                        local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                    )
                    local_features = torch.cat([local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1)
                    local_features = local_features.view(-1, n_dim2)

                    global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)
                else:
                    global_features = self._encode_views(image_ori)
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)
                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat([global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1)
                    global_features = global_features.view(-1, n_dim)
                    global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
                images_in_this_batch.append(global_local_features)

            if images_in_this_batch:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1), images_in_this_batch)
            idx += 1
        return inputs_embeds

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                images=None, images_seq_mask=None, images_spatial_crop=None, use_cache=None, cache_position=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, images, images_seq_mask, images_spatial_crop)
        return self.language_model.forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, images=None,
                                      images_seq_mask=None, images_spatial_crop=None, attention_mask=None,
                                      cache_position=None, **kwargs):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, cache_position=cache_position, **kwargs,
        )
        if cache_position is not None and cache_position[0] == 0:
            model_inputs["images"] = images
            model_inputs["images_seq_mask"] = images_seq_mask
            model_inputs["images_spatial_crop"] = images_spatial_crop
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.language_model._reorder_cache(past_key_values, beam_idx)

    def can_generate(self):
        return True

    def infer(self, tokenizer, prompt="", image_file="", output_path="", base_size=1024, image_size=640,
              crop_mode=True, test_compress=False, save_results=False, eval_mode=False, max_new_tokens=8192,
              no_repeat_ngram_size=35, ngram_window=128, temperature=0.0):
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f"{output_path}/images", exist_ok=True)

        if prompt and image_file:
            conversation = [
                {"role": "<|User|>", "content": f"{prompt}", "images": [f"{image_file}"]},
                {"role": "<|Assistant|>", "content": ""},
            ]
        elif prompt:
            conversation = [
                {"role": "<|User|>", "content": f"{prompt}"},
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            raise AssertionError("prompt is none!")

        prompt = format_messages(conversations=conversation, sft_format="plain", system_prompt="")

        patch_size = 16
        downsample_ratio = 4
        images = load_pil_images(conversation)
        valid_img_tokens = 0
        image_draw = images[0].copy() if images else None
        if image_draw is not None:
            w, h = image_draw.size
            ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
        else:
            w = h = 0
            ratio = 1

        image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        image_token = "<image>"  # nosec B105 - special token, not a secret
        text_splits = prompt.split(image_token)

        images_list, images_crop_list, images_seq_mask, tokenized_str, images_spatial_crop = [], [], [], [], []
        for text_sep, image in zip(text_splits, images):
            tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if crop_mode:
                if image.size[0] <= 640 and image.size[1] <= 640:
                    crop_ratio = [1, 1]
                else:
                    images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=image_size)
                global_view = ImageOps.pad(image, (base_size, base_size), color=tuple(int(x * 255) for x in image_transform.mean))
                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                images_list.append(image_transform(global_view))
                width_crop_num, height_crop_num = crop_ratio
                images_spatial_crop.append([width_crop_num, height_crop_num])
                if width_crop_num > 1 or height_crop_num > 1:
                    for cropped in images_crop_raw:
                        images_crop_list.append(image_transform(cropped))
                if image_size == 640:
                    valid_img_tokens += len(images_crop_list) * 100
                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
                tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
                tokenized_image += [IMAGE_TOKEN_ID]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([IMAGE_TOKEN_ID] * (num_queries * width_crop_num) + [IMAGE_TOKEN_ID]) * (num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
            else:
                if image_size <= 640:
                    image = image.resize((image_size, image_size))
                global_view = ImageOps.pad(image, (image_size, image_size), color=tuple(int(x * 255) for x in image_transform.mean))
                images_list.append(image_transform(global_view))
                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                images_spatial_crop.append([1, 1])
                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                tokenized_image = ([IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]) * num_queries
                tokenized_image += [IMAGE_TOKEN_ID]
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)

        tokenized_str += text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
        images_seq_mask += [False] * len(text_encode(tokenizer, text_splits[-1], bos=False, eos=False))
        tokenized_str = [0] + tokenized_str
        images_seq_mask = [False] + images_seq_mask

        input_ids = torch.LongTensor(tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, image_size, image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, base_size, base_size))
        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            images_crop = torch.stack(images_crop_list, dim=0) if images_crop_list else torch.zeros((1, 3, base_size, base_size))

        gen_kwargs = dict(
            images=[(images_crop, images_ori)],
            images_seq_mask=images_seq_mask.unsqueeze(0),
            images_spatial_crop=images_spatial_crop,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        # Match the original infer: use the sliding-window n-gram logits processor when a
        # window is given, otherwise fall back to HuggingFace's plain no_repeat_ngram_size.
        if no_repeat_ngram_size > 0 and ngram_window > 0:
            gen_kwargs["logits_processor"] = [SlidingWindowNoRepeatNgramProcessor(no_repeat_ngram_size, ngram_window)]
        elif no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if not eval_mode:
            gen_kwargs["streamer"] = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

        with torch.no_grad():
            output_ids = self.generate(input_ids.unsqueeze(0), **gen_kwargs)

        outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).shape[1]:])
        stop_str = "<｜end▁of▁sentence｜>"
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        if "<image>" in conversation[0]["content"] and eval_mode:
            return outputs

        if "<image>" in conversation[0]["content"] and test_compress and valid_img_tokens:
            pure_len = len(text_encode(tokenizer, outputs, bos=False, eos=False))
            print("=" * 50)
            print("image size: ", (w, h))
            print("valid image tokens: ", int(valid_img_tokens))
            print("output texts tokens (valid): ", pure_len)
            print("compression ratio: ", round(pure_len / valid_img_tokens, 2))
            print("=" * 50)

        if "<image>" in conversation[0]["content"] and save_results and image_draw is not None:
            print("=" * 15 + "save results:" + "=" * 15)
            matches_ref, matches_images, mathes_other = re_match(outputs)
            result = process_image_with_refs(image_draw, matches_ref, output_path)
            for i, a_match_image in enumerate(tqdm(matches_images, desc="image")):
                outputs = outputs.replace(a_match_image, "![](images/" + str(i) + ".jpg)\n")
            for a_match_other in tqdm(mathes_other, desc="other"):
                outputs = outputs.replace(a_match_other, "").replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
            with open(f"{output_path}/result.md", "w", encoding="utf-8") as afile:
                afile.write(outputs)
            result.save(f"{output_path}/result_with_boxes.jpg")

        return outputs
