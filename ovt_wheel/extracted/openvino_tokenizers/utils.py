# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import lru_cache
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import openvino
from openvino import Model, Output, Tensor, Type
from openvino import opset12 as opset
from openvino.op import Constant
from openvino.preprocess import PrePostProcessor

from .__version__ import __version__ as openvino_tokenizers_version
from .constants import (
    LOGITS_OUTPUT_NAME,
    ORIGINAL_POST_PROCESSOR_NAME,
    ORIGINAL_TOKENIZER_CLASS_NAME,
    PROCESSED_POST_PROCESSOR_NAME,
    TOKEN_IDS_OUTPUT_NAME,
    UTF8ReplaceMode,
    rt_info_to_hf_attribute_map,
)


@dataclass
class TokenzierConversionParams:
    """
    with_detokenizer : bool
        Whether to include a detokenizer in the conversion process. Default is False.

    add_special_tokens : bool
        Whether to add special tokens during tokenization. Default is True.

    skip_special_tokens : bool
        Whether to skip special tokens during detokenization. Default is True.

    clean_up_tokenization_spaces : Optional[bool]
        If True, extra spaces will be cleaned up during the tokenization process. Default is None.

    tokenizer_output_type : Type
        The output type for the tokenizer model. Default is `Type.i64`.

    detokenizer_input_type : Type
        The input type for the detokenizer model. Default is `Type.i64`.

    streaming_detokenizer : bool
        If True, enables streaming mode for the detokenizer. Default is False.

    use_max_padding : bool
        If True, enables maximum padding for the tokenizer. Default is False.

    max_length: Optional[int]
        The maximum length of the input sequence.

    handle_special_tokens_with_re : Optional[bool]
        If True, uses regular expressions to handle special tokens during tokenization. Default is None.

    use_sentencepiece_backend : bool
        If True, forces the use of the SentencePiece backend during tokenization. Default is False.

    utf8_replace_mode : Optional[UTF8ReplaceMode]
        Specifies the UTF-8 replacement mode during tokenization.
        Allowed values are UTF8ReplaceMode.DISABLE, UTF8ReplaceMode.IGNORE and UTF8ReplaceMode.REPLACE. Default is UTF8ReplaceMode.REPLACE.
    number_of_inputs: int
        The number of inputs for the model. Default is 1.
    """

    with_detokenizer: bool = False
    add_special_tokens: bool = True
    skip_special_tokens: bool = True
    clean_up_tokenization_spaces: Optional[bool] = None
    tokenizer_output_type: Type = Type.i64
    detokenizer_input_type: Type = Type.i64
    streaming_detokenizer: bool = False
    use_max_padding: bool = False
    max_length: Optional[int] = None
    handle_special_tokens_with_re: Optional[bool] = None
    use_sentencepiece_backend: bool = False
    utf8_replace_mode: Optional[UTF8ReplaceMode] = field(default_factory=lambda: UTF8ReplaceMode.REPLACE)
    add_attention_mask: bool = True
    add_prefix_space: Optional[bool] = None
    number_of_inputs: int = 1


logger = logging.getLogger(__name__)


def connect_models(
    first: Model,
    second: Model,
    name_map: Optional[Union[Sequence[tuple[str, str]], dict[str, str]]] = None,
    by_indices: bool = False,
    keep_second_model_unaligned_inputs: bool = True,
    keep_remaining_first_model_outputs: bool = False,
) -> Model:
    if by_indices:
        min_len = min(len(first.outputs), len(second.inputs))
        aligned_first_outputs = first.outputs[:min_len]
        aligned_second_inputs = second.inputs[:min_len]
    elif name_map is None:
        aligned_first_outputs = first.outputs
        aligned_second_inputs = [second.input(model1_output.get_any_name()) for model1_output in aligned_first_outputs]
    else:
        if isinstance(name_map, dict):
            name_map = list(name_map.items())
        aligned_first_outputs = [first.output(name1) for name1, _ in name_map]
        aligned_second_inputs = [second.input(name2) for _, name2 in name_map]

    for second_input, first_output in zip(aligned_second_inputs, aligned_first_outputs):
        logger.debug(f"Connecting: {first_output.get_any_name()} -> {second_input.get_any_name()}")
        for target in second_input.get_target_inputs():
            target.replace_source_output(first_output.get_node().input_value(0))
            # target.replace_source_output(model1_output)  # TODO: Produces incorrect topology

    new_inputs = first.inputs
    remaining_inputs = [input_ for input_ in second.inputs if input_ not in aligned_second_inputs]
    if keep_second_model_unaligned_inputs:
        new_inputs.extend(remaining_inputs)
    elif remaining_inputs:
        logger.info(
            "Some inputs of the second model were left uncovered and not included in the connected model: "
            + ", ".join(input_.name for input_ in remaining_inputs)
            + ". To add them set `keep_unaligned_inputs` to `True`"
        )
    new_inputs = [input_.get_node() for input_ in new_inputs]

    new_outputs = second.outputs
    remaining_outputs = [output for output in first.outputs if output not in aligned_first_outputs]
    if keep_remaining_first_model_outputs:
        new_outputs.extend(remaining_outputs)
    elif remaining_outputs:
        logger.info(
            "Some outputs of the first model were left uncovered and not included in the connected model: "
            + ", ".join(output.name for output in remaining_outputs)
            + ". To add them set `keep_unaligned_outputs` to `True`"
        )

    connected_model = Model(new_outputs, new_inputs, f"{first.get_name()}_with_{second.get_name()}")
    # TODO: Cleanup model1 and mode2 to avoid using them, they are ill-formed after the reconnection
    connected_model.validate_nodes_and_infer_types()
    return connected_model


def greedy_decoder(input) -> Model:
    argmax = opset.topk(
        data=input,
        k=1,
        axis=-1,
        mode="max",
        sort="none",
        name="ArgMax",
    )
    token_ids = opset.squeeze(
        data=argmax.output(1),
        axes=-1,
    )
    return token_ids.output(0)


def add_greedy_decoding(
    text_generation_model: Model, logits_output: str = LOGITS_OUTPUT_NAME, output_type: Type = Type.i64
) -> Model:
    ppp = PrePostProcessor(text_generation_model)
    ppp.output(logits_output).postprocess().custom(greedy_decoder)
    ppp.output(logits_output).tensor().set_element_type(output_type)
    model = ppp.build()
    model.output(logits_output).tensor.set_names({TOKEN_IDS_OUTPUT_NAME})
    return model


def change_inputs_type(model: Model, input_type: Type) -> Model:
    ppp = PrePostProcessor(model)
    for idx, _ in enumerate(model.inputs):
        ppp.input(idx).tensor().set_element_type(input_type)
    return ppp.build()


def change_outputs_type(model: Model, output_type: Type) -> Model:
    ppp = PrePostProcessor(model)
    for idx, _ in enumerate(model.outputs):
        ppp.output(idx).tensor().set_element_type(output_type)
    return ppp.build()


# from transformers.models.gpt2.tokenization_gpt2
@lru_cache()
def unicode_to_bytes() -> dict[str, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = (chr(n) for n in cs)
    return dict(zip(cs, bs))


def apply_unicode_to_bytes(token: str, return_corrupted_tokens: bool = False) -> bytes:
    bytes_encoder = unicode_to_bytes()
    try:
        return bytes(bytes_encoder[char] for char in token)
    except KeyError:
        # tokens that was not bytes-to-chars encoded
        # ModernBERT adds such tokens to the vocab directly, which is wrong, but we need to handle it
        if return_corrupted_tokens:
            return token.encode()
        return b""


def transform_unigram_token_to_bytes(token: str, byte_fallback: bool = False) -> bytes:
    token = token.replace("▁", " ")
    if byte_fallback and len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
        return bytes.fromhex(token[3:5])
    return token.encode()


def get_hf_tokenizer_attribute(
    hf_tokenizer: "PreTrainedTokenizerBase",  # noqa
    attributes: tuple[str],
) -> Any:
    return next((value for attr in attributes if (value := getattr(hf_tokenizer, attr, None)) is not None), None)


def get_package_version(name: str) -> str:
    import importlib.metadata as metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def update_rt_info_with_environment(ov_tokenizer: Model) -> None:
    """Adds package versions used for conversion to rt_info

    :param ov_tokenizer: Thes OpenVINO tokenizer model to update.
    :type ov_tokenizer: openvino.Model
    """
    ov_tokenizer.set_rt_info(openvino.get_version(), "openvino_version")
    ov_tokenizer.set_rt_info(openvino_tokenizers_version, "openvino_tokenizers_version")

    packages = ["transformers", "tiktoken", "sentencepiece", "tokenizers"]

    for name in packages:
        version = get_package_version(name)
        if version is not None:
            ov_tokenizer.set_rt_info(version, f"{name}_version")


def get_processor_template(
    hf_tokenizer: "PreTrainedTokenizerBase",  # noqa
) -> Optional[dict[str, Any]]:
    """Gets the JSON representation of the tokenizer post-processor template.

    :param hf_tokenizer: The Huggingface tokenizer object.
    :type hf_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
    :return: The JSON representation of the Huggingface tokenizer.
    :rtype: dict[str, Any]
    """
    if not getattr(hf_tokenizer, "is_fast", False):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_tokenizer.save_pretrained(tmpdir)
        try:
            with open(f"{tmpdir}/tokenizer.json", "r", encoding="utf-8") as f:
                tokenizer_json = json.load(f)
        except FileNotFoundError:
            return

    return tokenizer_json.get("post_processor", None)


def parse_template_processing(
    post_processor_json: dict[str, Any],
    hf_tokenizer: "PreTrainedTokenizerBase",  # noqa
) -> dict[str, dict[str, list[int]]]:
    vocab = hf_tokenizer.get_vocab()

    def parse_one_template(template: list[dict[str, Any]]) -> dict[str, list[int]]:
        ids, type_ids = [], []
        sequence_id = -1
        for element_type, element_dict in (next(iter(el.items())) for el in template):
            type_ids.append(element_dict["type_id"])
            if element_type == "Sequence":
                ids.append(sequence_id)
                sequence_id -= 1
            else:
                ids.append(vocab[element_dict["id"]])

        return {"ids": ids, "type_ids": type_ids}

    return {
        "single": parse_one_template(post_processor_json["single"]),
        "pair": parse_one_template(post_processor_json["pair"]),
    }


def parse_roberta_processing(
    post_processor_json: dict[str, Any],
    hf_tokenizer: "PreTrainedTokenizerBase",  # noqa
) -> dict[str, dict[str, list[int]]]:
    cls_id = post_processor_json["cls"][1]
    sep_id = post_processor_json["sep"][1]
    return {
        "single": {"ids": [cls_id, -1, sep_id], "type_ids": [0, 0, 0]},
        # roberta processor only uses 0 for type_id:
        # https://github.com/huggingface/tokenizers/blob/dd4fc3df1a8a7cd135eecca2158db018d85f94f1/tokenizers/src/processors/roberta.rs#L81
        "pair": {"ids": [cls_id, -1, sep_id, sep_id, -1, sep_id], "type_ids": [0, 0, 0, 0, 0, 0]},
    }


def parse_bert_processing(
    post_processor_json: dict[str, Any],
    hf_tokenizer: "PreTrainedTokenizerBase",  # noqa
) -> dict[str, dict[str, list[int]]]:
    cls_id = post_processor_json["cls"][1]
    sep_id = post_processor_json["sep"][1]
    return {
        "single": {"ids": [cls_id, -1, sep_id], "type_ids": [0, 0, 0]},
        "pair": {"ids": [cls_id, -1, sep_id, -1, sep_id], "type_ids": [0, 0, 0, 1, 1]},
    }


processor_parsers = {
    "TemplateProcessing": parse_template_processing,
    "RobertaProcessing": parse_roberta_processing,
    "BertProcessing": parse_bert_processing,
}


def parse_processor_template(
    post_processor_json: dict[str, Any],
    hf_tokenizer: "PreTrainedTokenizerBase",  # noqa
) -> Optional[dict[str, Any]]:
    if post_processor_json["type"] == "Sequence":
        post_processor_json = next(
            (processor for processor in post_processor_json["processors"] if processor["type"] in processor_parsers),
            {},
        )

    parser = processor_parsers.get(post_processor_json.get("type"), None)
    if parser is not None:
        return parser(post_processor_json, hf_tokenizer)


def update_rt_info_with_processor_template(
    ov_tokenizer: Model,
    hf_tokenizer: "PreTrainedTokenizerBase",  # noqa
) -> None:
    """Updates the rt_info of the tokenizer model with the post-processor template of the HF.

    Saves the original and the processed post-processor templates.
    Processed template uses negative ids for text inputs (A=-1, B=-2) and positive ids for special tokens.

    :param ov_tokenizer: The OpenVINO tokenizer model to update.
    :type ov_tokenizer: openvino.Model
    :param hf_tokenizer: The Huggingface tokenizer object.
    :type hf_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
    """
    post_processor_json = get_processor_template(hf_tokenizer)
    if post_processor_json is None:
        return

    ov_tokenizer.set_rt_info(json.dumps(post_processor_json), ORIGINAL_POST_PROCESSOR_NAME)
    parsed_post_processor = parse_processor_template(post_processor_json, hf_tokenizer)
    if parsed_post_processor is not None:
        ov_tokenizer.set_rt_info(json.dumps(parsed_post_processor), PROCESSED_POST_PROCESSOR_NAME)
    else:
        ov_tokenizer.set_rt_info(
            json.dumps(
                {"single": {"ids": [-1], "type_ids": [0]}, "pair": {"ids": [-1, -2], "type_ids": [0, 0]}}
            ),
            PROCESSED_POST_PROCESSOR_NAME
        )


def update_rt_info_with_params(
    ov_tokenizer: Model,
    hf_tokenizer: "PreTrainedTokenizerBase",  # noqa
    params: TokenzierConversionParams,
) -> None:
    """Updates the runtime information of the OpenVINO tokenizer model with the parameters and attributes of the HF.

    :param ov_tokenizer: The OpenVINO tokenizer model to update.
    :type ov_tokenizer: openvino.Model
    :param hf_tokenizer: The Huggingface tokenizer object.
    :type hf_tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerBase
    :param params: The conversion parameters.
    :type params: TokenzierConversionParams
    """
    ov_tokenizer.set_rt_info(str(type(hf_tokenizer)), ORIGINAL_TOKENIZER_CLASS_NAME)

    for key in fields(params):
        v = getattr(params, key.name)
        v = str(v) if isinstance(v, (bool, Enum)) else v
        ov_tokenizer.set_rt_info(v, key.name)

    for rt_field_name, hf_attributes in rt_info_to_hf_attribute_map.items():
        attribute = get_hf_tokenizer_attribute(hf_tokenizer, hf_attributes)
        if attribute is not None:
            ov_tokenizer.set_rt_info(attribute, rt_field_name)


def quote_meta(unquoted: Union[str, bytes]) -> str:
    if isinstance(unquoted, bytes):
        unquoted = unquoted.decode("latin1")
    symbols = []
    for char in unquoted:
        if not char.isalnum() and char not in ("_", "▁", "｜"):
            symbols.append("\\")
        symbols.append(char)
    return "".join(symbols)


def to_bytes(number: int) -> bytes:
    return number.to_bytes(4, "little")


def create_unpacked_string(strings: Iterable[str]) -> list[Output]:
    """
    Convert any list of strings to U8/1D numpy array with begins, ends, and chars
    """
    begins = BytesIO()
    ends = BytesIO()
    chars = BytesIO()
    offset = 0

    for string in strings:
        byte_string = string.encode("utf-8") if isinstance(string, str) else string
        length = len(byte_string)

        begins.write(to_bytes(offset))
        offset += length
        ends.write(to_bytes(offset))
        chars.write(byte_string)

    begins = np.frombuffer(begins.getvalue(), np.int32)
    ends = np.frombuffer(ends.getvalue(), np.int32)
    chars = np.frombuffer(chars.getvalue(), np.uint8)

    return [Constant(Tensor(x)).output(0) for x in [begins, ends, chars]]


def create_string_constant_node(value: Union[str, Iterable[str]]) -> list[Output]:
    if isinstance(value, str):
        # string scalar
        return Constant(np.frombuffer(bytes(value, "utf-8"), dtype=np.uint8)).outputs()
    elif isinstance(value, Iterable):
        # support only 1D strings for now
        return create_unpacked_string(value)
    else:
        raise ValueError(f"Unsupported value type {type(value)}")
