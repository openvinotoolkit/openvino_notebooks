import openvino as ov
import nncf
from pathlib import Path
import torch
import types
from typing import List, Optional, Tuple, Union, Callable
import openvino.opset13 as opset13
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
import numpy as np
import gc
from fireredtts2.fireredtts2 import FireRedTTS2
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import is_torch_xpu_available
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask, sdpa_mask
import shutil
import os
import json
from transformers import AutoTokenizer
import torchaudio
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
import re
import string
from tqdm import tqdm
import math
import torch.nn.functional as F


def patch_cos_sin_cached_fp32(model):
    if (
        hasattr(model, "layers")
        and hasattr(model.layers[0], "self_attn")
        and hasattr(model.layers[0].self_attn, "rotary_emb")
        and hasattr(model.layers[0].self_attn.rotary_emb, "dtype")
        and hasattr(model.layers[0].self_attn.rotary_emb, "inv_freq")
        and hasattr(model.layers[0].self_attn.rotary_emb, "max_position_embeddings")
        and hasattr(model.layers[0].self_attn.rotary_emb, "_set_cos_sin_cache")
    ):
        for layer in model.layers:
            if layer.self_attn.rotary_emb.dtype != torch.float32:
                layer.self_attn.rotary_emb._set_cos_sin_cache(
                    seq_len=layer.self_attn.rotary_emb.max_position_embeddings,
                    device=layer.self_attn.rotary_emb.inv_freq.device,
                    dtype=torch.float32,
                )


SYMBOLS_MAPPING = {
    "\n": "",
    "\t": "",
    "…": ",",
    "“": "",
    "”": "",
    "‘": "'",
    "’": "'",
    "【": "",
    "】": "",
    "[": "",
    "]": "",
    "（": "",
    "）": "",
    "(": "",
    ")": "",
    "・": "",
    "·": "",
    "「": "'",
    "」": "'",
    "《": "'",
    "》": "'",
    "—": "",
    "～": "，",
    "~": "，",
    "：": ",",
    "；": ",",
    ";": ",",
    ":": ",",
    '"': "",
    "！": "，",
    # "!": ".",
    "————": "",
    "——": "",
    "—": "",
    "……": "，",
    "*": "",
}

REPLACE_SYMBOL_REGEX = re.compile("|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys()))


EMOJI_REGEX = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "]+",
    flags=re.UNICODE,
)


def clean_text(text):
    # Clean the text
    text = text.strip()
    text = text.replace("\xa0", "")

    # Replace all chinese symbols with their english counterparts
    text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)

    # Remove emojis
    text = EMOJI_REGEX.sub(r"", text)

    # Remove continuous periods (...) and commas (,,,)
    text = re.sub(r"[.,]{2,}", lambda m: m.group()[0], text)

    return text


def utf_8_len(text):
    return len(text.encode("utf-8"))


def break_text(texts, length, splits: set):
    for text in texts:
        if utf_8_len(text) <= length:
            yield text
            continue

        curr = ""
        for char in text:
            curr += char

            if char in splits:
                yield curr
                curr = ""

        if curr:
            yield curr


def break_text_by_length(texts, length):
    for text in texts:
        if utf_8_len(text) <= length:
            yield text
            continue

        curr = ""
        for char in text:
            curr += char

            if utf_8_len(curr) >= length:
                yield curr
                curr = ""

        if curr:
            yield curr


def add_cleaned(curr, segments):
    curr = curr.strip()
    if curr and not all(c.isspace() or c in string.punctuation for c in curr):
        segments.append(curr)


def protect_float(text):
    # Turns 3.14 into <3_f_14> to prevent splitting
    return re.sub(r"(\d+)\.(\d+)", r"<\1_f_\2>", text)


def unprotect_float(text):
    # Turns <3_f_14> into 3.14
    return re.sub(r"<(\d+)_f_(\d+)>", r"\1.\2", text)


def split_text(text, length):
    text = clean_text(text)

    # Break the text into pieces with following rules:
    # 1. Split the text at ".", "!", "?" if text is NOT a float
    # 2. If the text is longer than length, split at ","
    # 3. If the text is still longer than length, split at " "
    # 4. If the text is still longer than length, split at any character to length

    texts = [text]
    texts = map(protect_float, texts)
    texts = break_text(texts, length, {".", "!", "?", "。", "！", "？"})
    texts = map(unprotect_float, texts)
    texts = break_text(texts, length, {",", "，"})
    texts = break_text(texts, length, {" "})
    texts = list(break_text_by_length(texts, length))

    # Then, merge the texts into segments with length <= length
    segments = []
    curr = ""

    for text in texts:
        if utf_8_len(curr) + utf_8_len(text) <= length:
            curr += text
        else:
            add_cleaned(curr, segments)
            curr = text

    if curr:
        add_cleaned(curr, segments)

    return segments


def contains_chinese(text):
    """检测文本是否包含中文字符"""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def count_words_english(text):
    """统计英文单词数量"""
    return len(text.split())


def count_characters_chinese(text):
    """统计中文字符数量"""
    return len(text)


def split_by_punctuation_english(text):
    """按英文标点符号分割"""
    sentences = re.split(r"([.!?])", text)
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if sentence:
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            result.append(sentence)

    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())

    return result


def split_by_punctuation_chinese(text):
    """按中文标点符号分割"""
    sentences = re.split(r"([。！？])", text)
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if sentence:
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            result.append(sentence)

    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())

    return result


def merge_sentences_english(sentences, max_words=80):
    """合并英文句子"""
    result = []
    current_chunk = ""

    for sentence in sentences:
        if not current_chunk:
            current_chunk = sentence
        else:
            test_chunk = current_chunk + " " + sentence
            if count_words_english(test_chunk) <= max_words:
                current_chunk = test_chunk
            else:
                result.append(current_chunk)
                current_chunk = sentence

    if current_chunk:
        result.append(current_chunk)

    return result


def merge_sentences_chinese(sentences, max_chars=100):
    """合并中文句子"""
    result = []
    current_chunk = ""

    for sentence in sentences:
        if not current_chunk:
            current_chunk = sentence
        else:
            test_chunk = current_chunk + sentence
            if count_characters_chinese(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                result.append(current_chunk)
                current_chunk = sentence

    if current_chunk:
        result.append(current_chunk)

    return result


def process_text(text):
    chinese_max_limit = 150
    english_max_limit = 80
    # 移除开头的标记如[S2]
    text = re.sub(r"^\[S\d+\]", "", text).strip()
    is_chinese = contains_chinese(text)
    if is_chinese:
        if count_characters_chinese(text) <= chinese_max_limit:
            return [text]
        sentences = split_by_punctuation_chinese(text)
        result = merge_sentences_chinese(sentences, chinese_max_limit)
    else:
        if count_words_english(text) <= english_max_limit:
            return [text]
        sentences = split_by_punctuation_english(text)
        result = merge_sentences_english(sentences, english_max_limit)

    return result


def process_text_list(text_list):
    new_text_list = []
    for text in text_list:
        speaker = text[:4]
        # print("---speaker:", speaker)
        assert speaker in ["[S1]", "[S2]", "[S3]", "[S4]"]
        result = process_text(text=text)
        # print("---result:\n", result, len(result))
        for chunk in result:
            new_text_list.append(speaker + chunk)
    return new_text_list


def _pad_and_chunk(audio: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    pad_len = math.ceil(audio.shape[1] / chunk_size) * chunk_size - audio.shape[1]
    audio = F.pad(audio, (0, pad_len), mode="constant", value=0)
    audio_chunks = audio.split(chunk_size, dim=1)
    return audio_chunks


def _multinomial_sample_one_no_sync(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def prepare_padding_mask(attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int, _slice: bool = True) -> Optional[torch.Tensor]:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
    according to the `kv_offset` if `_slice` is `True`.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # Pad it if necessary
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        # For flex, we should not slice them, only use an offset
        if _slice:
            # Equivalent to: `local_padding_mask = attention_mask[:, kv_offset : kv_offset + kv_length]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indices = torch.arange(kv_length, device=local_padding_mask.device)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask


def and_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_function as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def _ignore_causal_mask_sdpa(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """
    Detects whether the causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the 2D `padding_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    is_tracing = torch.jit.is_tracing() or isinstance(padding_mask, torch.fx.Proxy) or is_torchdynamo_compiling()
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    # When using `torch.export` or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is
    # hard-coded to the forward. If a user exports a model with query_length > 1, the exported model will hard-code `is_causal=True`
    # which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108). Thus, we only set
    # `ignore_causal_mask = True` if we are not tracing
    if (
        not is_tracing
        # only cases when lower and upper diags are the same, see https://github.com/pytorch/pytorch/issues/108108
        and (query_length == 1 or (kv_length == query_length or is_torch_xpu_available))
        # in this case we need to add special patterns to the mask so cannot be skipped otherwise
        and (local_attention_size is None or kv_length < local_attention_size)
        # In this case, we need to add padding to the mask, so cannot be skipped otherwise
        and (padding_mask is None or (padding_mask.all() if not is_torch_xpu_available or query_length == 1 else padding_mask[:, :query_length].all()))
    ):
        return True

    return False


def sdpa_mask_without_vmap(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Optional[Callable] = None,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    if mask_function is None:
        mask_function = causal_mask_function

    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # Create broadcatable indices
    device = cache_position.device
    q_indices = cache_position[None, None, :, None]
    head_indices = torch.arange(1, dtype=torch.long, device=device)[None, :, None, None]
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None, None, None]
    kv_indices = torch.arange(kv_length, dtype=torch.long, device=device)[None, None, None, :] + kv_offset

    # Apply mask function element-wise through broadcasting
    causal_mask = mask_function(batch_indices, head_indices, q_indices, kv_indices)
    # Expand the mask to match batch size and query length if they weren't used in the mask function
    causal_mask = causal_mask.expand(batch_size, -1, q_length, kv_length)

    return causal_mask


# Adapted from https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/masking_utils.py#L433
# Specifically for OpenVINO, we use torch.finfo(torch.float16).min instead of torch.finfo(dtype).min
def eager_mask_without_vmap(*args, **kwargs) -> Optional[torch.Tensor]:
    kwargs.pop("allow_is_causal_skip", None)
    dtype = kwargs.get("dtype", torch.float32)
    mask = sdpa_mask_without_vmap(*args, allow_is_causal_skip=False, **kwargs)
    # we use torch.finfo(torch.float16).min instead torch.finfo(dtype).min to avoid an overflow but not
    # sure this is the right way to handle this, we are basically pretending that -65,504 is -inf
    mask = torch.where(
        mask,
        torch.tensor(0.0, device=mask.device, dtype=dtype),
        torch.tensor(torch.finfo(torch.float16).min, device=mask.device, dtype=dtype),
    )
    return mask


# for OpenVINO, we use torch.finfo(torch.float16).min instead of torch.finfo(dtype).min
# Although I'm not sure this is the right way to handle this, we are basically pretending that -65,504 is -inf
ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask_without_vmap)

# for decoder models, we use eager mask without vmap for sdpa as well
# to avoid a nan output issue in OpenVINO that only happens in case of:
# non-stateful models on cpu and stateful models on npu
ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", eager_mask_without_vmap)


def model_has_state(ov_model: ov.Model):
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`list[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`list[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
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


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    key_value_output_names: list[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`list[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`list[str]`):
            list of names for key value input layers
        key_value_output_names (`list[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
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
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model, dim=1):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[2:-1]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[dim:]]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )


core = ov.Core()


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


TEXT_EMBEDDINGS_PATH = "openvino_text_embeddings_model.xml"
AUDIO_EMBEDDINGS_PATH = "openvino_audio_embeddings_model.xml"
AUDIO_DECODER_PATH = "openvino_audio_decoder_model.xml"
AUDIO_UPSAMPLER_PATH = "openvino_audio_upsampler_model.xml"
AUDIO_ENCODER_PATH = "openvino_audio_encoder_model.xml"
DECODER_MODEL_PATH = "openvino_decoder_model.xml"
BACKBONE_MODEL_PATH = "openvino_backbone_model.xml"


def convert_fireredtts2(model_id, model_path=None, quantization_config=None):
    if model_path is None:
        model_path = Path(model_id.split("/")[-1])
    else:
        model_path = Path(model_path)

    if all(
        (model_path / model_name).exists()
        for model_name in [
            TEXT_EMBEDDINGS_PATH,
            AUDIO_DECODER_PATH,
            AUDIO_ENCODER_PATH,
            AUDIO_EMBEDDINGS_PATH,
            DECODER_MODEL_PATH,
            BACKBONE_MODEL_PATH,
            AUDIO_UPSAMPLER_PATH,
        ]
    ):
        print(f"✅ {model_id} model already converted. You can find results in {model_path}")
        return model_path
    print(f"⌛ {model_id} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    pt_model = FireRedTTS2(
        pretrained_dir=model_id,
        gen_type="dialogue",
        device="cpu",
    )

    print("✅ Original model successfully loaded")
    print("⌛ Export tokenizer and config")

    pt_model._text_tokenizer.save_pretrained(model_path)
    for json_file in Path(model_id).glob("*.json"):
        shutil.copy(json_file, model_path / json_file.name)

    if not (model_path / TEXT_EMBEDDINGS_PATH).exists():
        print("⌛ Convert TEXT_EMBEDDINGS model")

        ov_model = ov.convert_model(pt_model._model.text_embeddings, example_input=torch.ones([1, 1], dtype=torch.int32))
        ov.save_model(ov_model, model_path / TEXT_EMBEDDINGS_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ TEXT_EMBEDDINGS model successfully converted")

    if not (model_path / AUDIO_EMBEDDINGS_PATH).exists():
        print("⌛ Convert AUDIO_EMBEDDINGS model")

        ov_model = ov.convert_model(pt_model._model.audio_embeddings, example_input=torch.ones([10], dtype=torch.int32))
        ov.save_model(ov_model, model_path / AUDIO_EMBEDDINGS_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ AUDIO_EMBEDDINGS model successfully converted")

    if not (model_path / AUDIO_UPSAMPLER_PATH).exists():
        print("⌛ Convert AUDIO_UPSAMPLER model")

        def forward_wrap_audio_upsampler(self, tokens: torch.Tensor):
            tokens = tokens.permute(1, 0, 2)  # (B, nq, L) -> (nq, B, L)
            vq_out_feats = self.rvq.decode_codes(tokens)
            vq_out_feats = vq_out_feats.transpose(1, 2)
            vq_out_length = torch.tensor([vq_out_feats.size(1)], dtype=torch.long, device=vq_out_feats.device)
            vq_out_feats, vq_out_length = self.upsample(vq_out_feats, vq_out_length)
            return vq_out_feats, vq_out_length

        pt_model._audio_tokenizer._orig_forward = pt_model._audio_tokenizer.forward
        pt_model._audio_tokenizer.forward = types.MethodType(forward_wrap_audio_upsampler, pt_model._audio_tokenizer)

        ov_model = ov.convert_model(pt_model._audio_tokenizer, example_input=torch.ones([1, 16, 1], dtype=torch.int32))
        ov.save_model(ov_model, model_path / AUDIO_UPSAMPLER_PATH)
        del ov_model
        pt_model._audio_tokenizer.forward = pt_model._audio_tokenizer._orig_forward
        del pt_model._audio_tokenizer._orig_forward
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ AUDIO_UPSAMPLER model successfully converted")

    if not (model_path / AUDIO_DECODER_PATH).exists():
        print("⌛ Convert AUDIO_DECODER model")
        example_input = {
            "x": torch.ones([1, 584, 768], dtype=torch.float32),
            "x_lens": torch.tensor([584], dtype=torch.int64),
        }

        ov_model = ov.convert_model(pt_model._audio_tokenizer.acoustic_decoder, example_input=example_input)
        ov.save_model(ov_model, model_path / AUDIO_DECODER_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ AUDIO_DECODER model successfully converted")

    if not (model_path / AUDIO_ENCODER_PATH).exists():
        print("⌛ Convert AUDIO_ENCODER model")

        def forward_wrap_audio_encoder(self, audio16k: torch.Tensor):
            return self._encode_one_batch(audio16k)

        pt_model._audio_tokenizer._orig_forward = pt_model._audio_tokenizer.forward
        pt_model._audio_tokenizer.forward = types.MethodType(forward_wrap_audio_encoder, pt_model._audio_tokenizer)

        ov_model = ov.convert_model(pt_model._audio_tokenizer, example_input=torch.ones([1, 96000], dtype=torch.float32))
        ov.save_model(ov_model, model_path / AUDIO_ENCODER_PATH)
        del ov_model
        pt_model._audio_tokenizer.forward = pt_model._audio_tokenizer._orig_forward
        del pt_model._audio_tokenizer._orig_forward
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ AUDIO_ENCODER model successfully converted")

    if not (model_path / DECODER_MODEL_PATH).exists():
        print("⌛ Convert DECODER_MODEL model")
        patch_cos_sin_cached_fp32(pt_model._model.decoder)
        if hasattr(pt_model._model.decoder, "model"):
            patch_cos_sin_cached_fp32(pt_model._model.decoder.model)

        def forward_wrap_decoder(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            step: Optional[torch.Tensor] = None,
        ):
            if past_key_values is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            inputs_embeds_proj = self.projection(inputs_embeds)
            # print(f"decoder inputs: {inputs_embeds}")
            outputs = self.decoder(
                attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds_proj, use_cache=True
            )

            if past_key_values is not None:
                outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()
            decoder_h = outputs.last_hidden_state
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[step - 1])
            return (ci_logits, outputs.past_key_values)

        num_pkv = pt_model._model.decoder.config.num_hidden_layers
        hidden_size = pt_model._model.decoder.config.hidden_size

        pt_model._model._orig_forward = pt_model._model.forward
        pt_model._model.forward = types.MethodType(forward_wrap_decoder, pt_model._model)

        pkv_shape = (
            2,
            pt_model._model.decoder.config.num_key_value_heads,
            2,
            pt_model._model.decoder.config.hidden_size // pt_model._model.decoder.config.num_attention_heads,
        )

        inputs_embeds = torch.randn((2, 2, hidden_size))
        attention_mask = torch.ones([2, 4], dtype=torch.int64)
        position_ids = torch.arange(2).unsqueeze(0).expand(2, -1)

        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits"]
        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.extend(["inputs_embeds"])
        example_input = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "step": torch.tensor(1).to(dtype=torch.int32),
        }

        input_shapes = [
            ov.PartialShape([-1, -1]),  # attention_mask
            ov.PartialShape([-1, -1]),  # position_ids (2D for code predictor)
        ]
        input_shapes += (
            [
                ov.PartialShape(
                    [
                        -1,
                        pt_model._model.decoder.config.num_key_value_heads,
                        -1,
                        pt_model._model.decoder.config.hidden_size // pt_model._model.decoder.config.num_attention_heads,
                    ]
                )
            ]
            * 2
            * num_pkv
        )
        input_shapes += [ov.PartialShape([-1, -1, hidden_size]), ov.PartialShape([])]  # inputs_embeds
        __make_16bit_traceable(pt_model._model)

        ov_model = ov.convert_model(pt_model._model, example_input=example_input, input=input_shapes)
        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})
        patch_stateful(ov_model)
        print("✅ Decoder model successfully converted")
        if quantization_config is not None and "llm" in quantization_config:
            print(f"⌛ Weights compression with {quantization_config['llm']['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config["llm"])
            print("✅ Weights compression finished")
        else:
            ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
        ov.save_model(ov_model, model_path / DECODER_MODEL_PATH)
        del ov_model
        pt_model._model.forward = pt_model._model._orig_forward
        del pt_model._model._orig_forward
        cleanup_torchscript_cache()
        gc.collect()

    if not (model_path / BACKBONE_MODEL_PATH).exists():
        print("⌛ Convert BACKBONE_MODEL model")

        patch_cos_sin_cached_fp32(pt_model._model.backbone)
        if hasattr(pt_model._model.backbone, "model"):
            patch_cos_sin_cached_fp32(pt_model._model.backbone.model)

        backbone_config = pt_model._model.backbone.config
        backbone_config.save_pretrained(model_path)

        def forward_wrap_backbone(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
        ):
            if past_key_values is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            outputs = self.backbone(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=True
            )
            if past_key_values is not None:
                outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()
            h = outputs.last_hidden_state
            last_h = h[:, -1, :]
            c0_logits = self.codebook0_head(last_h)
            output = (c0_logits, last_h, outputs.past_key_values)
            return output

        num_pkv = pt_model._model.backbone.config.num_hidden_layers
        hidden_size = pt_model._model.backbone.config.hidden_size
        pt_model._model._orig_forward = pt_model._model.forward
        pt_model._model.forward = types.MethodType(forward_wrap_backbone, pt_model._model)
        pkv_shape = (
            2,
            pt_model._model.backbone.config.num_key_value_heads,
            2,
            pt_model._model.backbone.config.hidden_size // pt_model._model.backbone.config.num_attention_heads,
        )

        input_embeds = torch.randn((2, 2, hidden_size))
        attention_mask = torch.ones([2, 4], dtype=torch.int64)
        position_ids = torch.arange(2).unsqueeze(0).expand(2, -1)

        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits", "last_hidden_state"]
        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.extend(["inputs_embeds"])

        example_input = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": input_embeds,
        }

        input_shapes = [
            ov.PartialShape([-1, -1]),
            ov.PartialShape([-1, -1]),
        ]
        input_shapes += (
            [
                ov.PartialShape(
                    [
                        -1,
                        pt_model._model.backbone.config.num_key_value_heads,
                        -1,
                        pt_model._model.backbone.config.hidden_size // pt_model._model.backbone.config.num_attention_heads,
                    ]
                )
            ]
            * 2
            * num_pkv
        )
        input_shapes += [ov.PartialShape([-1, -1, hidden_size])]  # inputs_embeds

        __make_16bit_traceable(pt_model._model)
        ov_model = ov.convert_model(pt_model._model, example_input=example_input, input=input_shapes)
        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})
        patch_stateful(ov_model, 2)
        print("✅ Backbone model successfully converted")
        if quantization_config is not None and "llm" in quantization_config:
            print(f"⌛ Weights compression with {quantization_config['llm']['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config["llm"])
            print("✅ Weights compression finished")
        else:
            ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
        ov.save_model(ov_model, model_path / BACKBONE_MODEL_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
    del pt_model
    gc.collect()
    print(f"✅ {model_id} model conversion finished. You can find results in {model_path}")
    return model_path


@dataclass
class Segment:
    speaker: str
    text: str
    audio: torch.Tensor


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    decoder_loss_weight: float
    use_text_loss: bool


class OVFireRedTTS2:
    def __init__(self, pretrained_dir, gen_type, device, codec_device="CPU"):
        self.device = device
        self.codec_device = codec_device
        self.sample_rate = 16000
        self.max_seq_len = 3100

        assert os.path.exists(pretrained_dir)
        assert gen_type in ["monologue", "dialogue"]
        llm_config_path = os.path.join(pretrained_dir, "config_llm.json")
        codec_config_path = os.path.join(pretrained_dir, "config_codec.json")

        # check
        assert os.path.exists(llm_config_path)
        assert os.path.exists(codec_config_path)

        # ==== Load Torch LLM ====
        llm_config = json.load(open(llm_config_path))
        self.config = ModelArgs(
            backbone_flavor=llm_config["llm_models"]["backbone_flavor"],
            decoder_flavor=llm_config["llm_models"]["decoder_flavor"],
            text_vocab_size=llm_config["llm_models"]["text_vocab_size"],
            audio_vocab_size=llm_config["llm_models"]["audio_vocab_size"],
            audio_num_codebooks=llm_config["llm_models"]["audio_num_codebooks"],
            decoder_loss_weight=llm_config["llm_models"]["decoder_loss_weight"],
            use_text_loss=True,
        )

        model_dir = Path(pretrained_dir)
        self.backbone = core.compile_model(model_dir / BACKBONE_MODEL_PATH, self.device).create_infer_request()
        self.decoder = core.compile_model(model_dir / DECODER_MODEL_PATH, self.device).create_infer_request()
        self.audio_embeddings = core.compile_model(model_dir / AUDIO_EMBEDDINGS_PATH, self.device)
        self.audio_decoder = core.compile_model(model_dir / AUDIO_DECODER_PATH, self.codec_device)
        self.audio_encoder = core.compile_model(model_dir / AUDIO_ENCODER_PATH, self.codec_device)
        self.text_embeddings = core.compile_model(model_dir / TEXT_EMBEDDINGS_PATH, self.device)
        self.audio_upsampler = core.compile_model(model_dir / AUDIO_UPSAMPLER_PATH, self.device)
        print("[INFO] OV model Loaded...")

        # ==== Load Qwen2.5 Text Tokenizer ====
        self._text_tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
        print("[INFO] Text Tokenizer Loaded...")

    def encode(
        self,
        audio16k: torch.Tensor,
        audio16k_length: torch.Tensor = None,
        batch_size: int = 96,
    ):
        """
        Args:
            audio16k: shape (b, t)
            audio16k_length: (b,)
        Returns:
            token: shape (b, nq, l)
            token_length: (b,)
        """
        if audio16k_length is None:
            assert audio16k.shape[0] == 1
            audio16k_length = torch.tensor([audio16k.shape[1]], dtype=torch.long, device=audio16k.device)

        CHUNK_SIZE = 6 * 16000
        B, T = audio16k.shape
        # Pad, chunk, and batch
        audio16k_batch = []
        batch_size_list = []
        for i in range(B):
            # Remove extra paddings
            one_audio_chunks = _pad_and_chunk(audio16k[i : (i + 1), : audio16k_length[i]], CHUNK_SIZE)
            audio16k_batch += one_audio_chunks
            batch_size_list.append(len(one_audio_chunks))
        audio16k_batch = torch.cat(audio16k_batch, dim=0)
        # Batch encode
        token_batch = []
        for i in range(0, audio16k_batch.shape[0], batch_size):
            one_audio_batch = audio16k_batch[i : (i + batch_size)]
            one_token_batch = torch.from_numpy(self.audio_encoder(one_audio_batch)[0])
            token_batch.append(one_token_batch)
        token_batch = torch.cat(token_batch, dim=0)
        # Recover & concat
        token_list = torch.split(token_batch, batch_size_list, dim=0)  # [(B=1, nq, l), (B=3, nq, l), ...]
        token_list = [torch.cat(token_ts.split(1, dim=0), dim=-1) for token_ts in token_list]  # (B=1, nq, l)
        # Pad tokens
        token = pad_sequence(
            [ts.squeeze(0).transpose(1, 0) for ts in token_list],
            batch_first=True,
            padding_value=0,
        ).transpose(
            1, 2
        )  # (B, nq, L)
        token_length = (audio16k_length / 1280).ceil().long()
        token = token[..., : token_length.max()]  # Remove extra paddings (we pad to multiples of 6s)
        return token, token_length

    def load_prompt_audio(self, audio_path) -> torch.Tensor:
        audio, audio_sr = torchaudio.load(audio_path)
        # Audio must be single channel
        if audio.shape[0] > 1:
            audio = audio[0, :].unsqueeze(0)
        audio16k = torchaudio.functional.resample(audio, audio_sr, 16000)
        return audio16k

    def prepare_prompt(self, text, speaker, audio_path) -> Segment:
        audio_tensor = self.load_prompt_audio(audio_path)
        return Segment(text=text, speaker=speaker, audio=audio_tensor)

    def _tokenize_text_segment(self, text: str, speaker: str) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text = speaker + "<|text_start|>" + text + "<|text_end|>"
        text_tokens = self._text_tokenizer.encode(text)
        text_frame = torch.zeros(len(text_tokens), 17).long()
        text_frame_mask = torch.zeros(len(text_tokens), 17).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame)
        frame_masks.append(text_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio_length = torch.tensor([audio.shape[1]], dtype=torch.long)
        audio_tokens, token_length = self.encode(
            audio,
            audio_length,
            batch_size=48,
        )

        audio_tokens = audio_tokens.squeeze(0)
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 17).long()
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 17).bool()
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len,17), (seq_len, 17)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """

        # assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        backbone_attention_mask = torch.ones(tokens.size(0), tokens.size(1), dtype=torch.long, device=tokens.device)  # [batch, curr_seq_len]
        backbone_position_ids = input_pos
        if self.backbone_past_len != 0:
            backbone_attention_mask = torch.cat(
                [torch.ones(tokens.size(0), self.backbone_past_len, dtype=torch.long, device=tokens.device), backbone_attention_mask], dim=1
            )
            backbone_position_ids = backbone_position_ids[:, -tokens.shape[1] :]

        inputs = {
            "inputs_embeds": h,
            "attention_mask": backbone_attention_mask,
            "position_ids": backbone_position_ids,
            "beam_idx": np.arange(h.shape[0], dtype=int),
        }

        self.backbone.start_async(inputs, share_inputs=True)
        self.backbone.wait()
        logits = self.backbone.get_tensor("logits").data
        last_hidden_state = self.backbone.get_tensor("last_hidden_state").data
        c0_logits = torch.from_numpy(logits)
        last_h = torch.from_numpy(last_hidden_state)
        self.backbone_past_len += inputs["inputs_embeds"].shape[1]
        c0_sample = sample_topk(c0_logits, 1, temperature)
        c0_embed = self._embed_audio(0, c0_sample)
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        self.decoder.reset_state()
        # Set initial value for the next beam_idx input that will be used at the current iteration
        # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
        decoder_past_length = 0
        for i in range(1, self.config.audio_num_codebooks):
            decoder_attention_mask = torch.ones(curr_h.size(0), curr_h.size(1), dtype=torch.long, device=curr_h.device)  # [batch, curr_seq_len]
            decoder_position_ids = curr_pos  # [batch, curr_seq_len]
            if decoder_past_length != 0:
                decoder_attention_mask = torch.cat(
                    [torch.ones(curr_h.size(0), decoder_past_length, dtype=torch.long, device=curr_h.device), decoder_attention_mask], dim=1
                )
                decoder_position_ids = decoder_position_ids[:, -curr_h.shape[1] :]

            inputs = {
                "inputs_embeds": curr_h,
                "attention_mask": decoder_attention_mask,
                "position_ids": decoder_position_ids,
                "beam_idx": np.arange(curr_h.shape[0], dtype=int),
                "step": torch.tensor(i).to(dtype=torch.int32),
            }

            self.decoder.start_async(inputs, share_inputs=True)
            self.decoder.wait()
            logits = self.decoder.get_tensor("logits").data
            ci_logits = torch.from_numpy(logits)
            decoder_past_length += inputs["inputs_embeds"].shape[1]
            ci_sample = sample_topk(ci_logits, 1, 0.75)  # fix to 10 and 0.75
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        self.backbone.past_key_values = None
        self.decoder.past_key_values = None

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.audio_embeddings((tokens + codebook * self.config.audio_vocab_size)[0])[0]).unsqueeze(0)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = torch.from_numpy(self.text_embeddings(tokens[:, :, -1])[0]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks, device=tokens.device))
        audio_embeds = torch.from_numpy(self.audio_embeddings(audio_tokens.view(-1))[0]).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def generate(
        self,
        text: str,
        speaker: str,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 20,
    ) -> torch.Tensor:
        self.backbone.reset_state()
        self.backbone_past_len = 0
        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long()
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool()

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long()

        max_seq_len = 3100
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}")

        for _ in range(max_generation_len):
            sample = self.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            # eos
            if torch.all(sample == 0):
                break

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long()], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [
                    torch.ones_like(sample).bool(),
                    torch.zeros(1, 1).bool(),
                ],
                dim=1,
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
        vq_out = self.audio_upsampler(torch.stack(samples).permute(1, 2, 0))
        vq_out_feats, _ = torch.from_numpy(vq_out[0]), torch.from_numpy(vq_out[1])
        vq_out_length = torch.tensor([vq_out_feats.shape[1]], dtype=torch.long)
        audio = torch.from_numpy(self.audio_decoder([vq_out_feats, vq_out_length])[0])
        audio = audio.squeeze(0).squeeze(0)

        return audio

    @torch.inference_mode()
    def generate_dialogue(
        self,
        text_list,
        prompt_wav_list=None,
        prompt_text_list=None,
        temperature=0.9,
        topk=20,
    ):
        all_generated_segments = []
        all_storage_segments = []
        prompt_segments = []
        text_list = process_text_list(text_list=text_list)
        if prompt_wav_list is not None:
            assert len(prompt_wav_list) == len(prompt_text_list)
            # Prepare prompts
            for i in range(len(prompt_wav_list)):
                prompt_wav = prompt_wav_list[i]
                prompt_text = prompt_text_list[i]
                speaker = prompt_text[:4]
                assert speaker in ["[S1]", "[S2]", "[S3]", "[S4]"]
                prompt_segments.append(self.prepare_prompt(text=prompt_text, speaker=speaker, audio_path=prompt_wav))

        for text in tqdm(text_list):
            speaker = text[:4]
            text = text[4:]
            # print("---speaker:", speaker)
            # print("---text:", text)
            assert speaker in ["[S1]", "[S2]", "[S3]", "[S4]"]

            audio_tensor = self.generate(
                text=text,
                speaker=speaker,
                context=prompt_segments + all_generated_segments,
                max_audio_length_ms=30_000,
                temperature=temperature,
                topk=topk,
            )

            # 做上下文管理的时候需要将audio 转到16k
            audio_16k = torchaudio.functional.resample(audio_tensor.unsqueeze(0), 24000, 16000)
            all_generated_segments.append(Segment(text=text, speaker=speaker, audio=audio_16k))

            all_storage_segments.append(Segment(text=text, speaker=speaker, audio=audio_tensor.unsqueeze(0)))

        # Concatenate all generations
        all_audio = torch.cat([seg.audio for seg in all_storage_segments], dim=1)
        all_audio = all_audio.cpu()
        return all_audio
