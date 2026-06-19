# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import logging
import weakref
from collections import defaultdict
from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass, field
from functools import reduce, singledispatchmethod
from itertools import groupby, islice
from operator import add
from typing import Any, Optional, Union

import numpy as np
from openvino import Model, Output, PartialShape, Shape, Type, op
from openvino import opset12 as opset
from openvino.exceptions import OVTypeError, UserInputError
from openvino.utils.types import as_node, make_constant_node

from . import _get_factory, _get_opset_factory
from .constants import (
    ATTENTION_MASK_INPUT_NAME,
    DETOKENIZER_NAME,
    MIN_CACHE_CAPACITY,
    STRING_OUTPUT_NAME,
    TOKEN_IDS_INPUT_NAME,
    TOKEN_TYPE_IDS_INPUT_NAME,
    TOKENIZER_NAME,
    VOCAB_SIZE_CACHE_PROPORTION,
    UTF8ReplaceMode,
)
from .utils import (
    apply_unicode_to_bytes,
    create_string_constant_node,
    quote_meta,
    transform_unigram_token_to_bytes,
)


logger = logging.getLogger(__name__)


@dataclass
class BasePipelineStep:
    _pipeline: Optional[weakref.ReferenceType["TokenizerPipeline"]] = field(default=None, init=False, repr=False)

    def __str__(self) -> str:
        params_string = ", ".join(f"{key}={val!r}" for key, val in self.get_config().items())
        return f"{self.__class__.__name__}({params_string})"

    def get_config(self) -> dict[str, Any]:
        config = {key: value for key, value in vars(self).items() if not key.startswith("_")}
        properties = {
            key: getattr(self, key)
            for key in dir(type(self))
            if not key.startswith("_") and isinstance(getattr(type(self), key), property)
        }
        config.update(properties)
        return config

    def get_pipeline(self) -> Optional["TokenizerPipeline"]:
        return self._pipeline() if self._pipeline is not None else None

    def set_pipeline(self, pipeline: "TokenizerPipeline") -> None:
        self._pipeline = weakref.ref(pipeline)

    def get_ov_subgraph(self, *input_nodes: list[Output]) -> list[Output]:
        raise NotImplementedError

    def finalize(self) -> None:
        """Called after the entire pipeline has been built"""
        return


@dataclass(frozen=True, order=True)
class SpecialToken:
    text: str
    strip_left: bool = False
    strip_right: bool = False

    def regex_repr(self) -> str:
        # operation has to be rewritten with RE2:Set in order to support multiple
        return r"(?:\s*)" * self.strip_left + f"({quote_meta(self.text)})" + r"(?:\s*)" * self.strip_right


@dataclass
class SpecialTokensSplit(BasePipelineStep):
    special_tokens: list[SpecialToken] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, reverse=True)

    @classmethod
    def from_string_iter(cls, strings: Iterable[str]) -> "SpecialTokensSplit":
        return cls([SpecialToken(string) for string in strings])

    @classmethod
    def from_hf_tokenizer(cls, hf_tokenizer) -> "SpecialTokensSplit":
        added_tokens = {}

        if hasattr(hf_tokenizer, "special_tokens"):
            added_tokens.update(
                {
                    idx: SpecialToken(token)
                    for token, idx in sorted(hf_tokenizer.special_tokens.items(), key=lambda x: x[1])
                }
            )
            # if padding and unk tokens share the same index, use unk
            if hf_tokenizer.unk_token is not None and hf_tokenizer.unk_token not in (
                token.text for token in added_tokens.values()
            ):
                unk_token_id = hf_tokenizer.unk_token_id or hf_tokenizer.pad_token_id
                added_tokens[unk_token_id] = hf_tokenizer.unk_token

        if hasattr(hf_tokenizer, "tokenizer") and hasattr(hf_tokenizer.tokenizer, "index_special_tokens"):
            added_tokens.update(hf_tokenizer.tokenizer.index_special_tokens)

        if added_tokens_decoder := getattr(hf_tokenizer, "added_tokens_decoder", False):
            added_tokens.update(
                {
                    idx: SpecialToken(
                        text=added_token.content,
                        strip_left=added_token.lstrip,
                        strip_right=added_token.rstrip,
                    )
                    for idx, added_token in added_tokens_decoder.items()
                }
            )

        return cls(special_tokens=list(added_tokens.values()))

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        if not self.special_tokens:
            return list(input_nodes)

        grouped_tokens = defaultdict(list)

        for token in self.special_tokens:
            grouped_tokens[(token.strip_left, token.strip_right)].append(token)

        split_pattern = "|".join(
            (
                r"(?:\s*)" * strip_left
                + "("
                + "|".join(quote_meta(token.text) for token in tokens)
                + ")"
                + r"(?:\s*)" * strip_right
            )
            for (strip_left, strip_right), tokens in grouped_tokens.items()
        )
        input_nodes.extend(create_string_constant_node(split_pattern))

        return _get_factory().create("SpecialTokensSplit", input_nodes).outputs()


@dataclass
class NormalizationStep(BasePipelineStep):
    pass


@dataclass
class NormalizeUnicode(NormalizationStep):
    normalization_form: str = "NFD"

    def __post_init__(self):
        if self.normalization_form not in ["NFD", "NFC", "NFKD", "NFKC"]:
            raise ValueError(
                '[ NormalizeUnicode ] `normalization_form` attribute must be one of ["NFD", "NFC", "NFKD", "NFKC"], '
                f"got {self.normalization_form}."
            )

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        return (
            _get_factory()
            .create(
                "CharsMapNormalization",
                input_nodes,
                {
                    "normalization_form": self.normalization_form.lower(),
                    "remove_extra_whitespaces": False,
                },
            )
            .outputs()
        )


@dataclass
class CaseFoldStep(NormalizationStep):
    #  attribute from tf.StringLower operation
    encoding: str = "utf-8"

    def __post_init__(self):
        if self.encoding not in ["", "utf-8"]:
            raise ValueError(
                f"[ CaseFoldStep ] `encoding` attribute must be one of ['', 'utf-8'], got {self.encoding!r}."
            )

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        if self.encoding == "":
            return _get_factory().create("CaseFold", input_nodes, {"encoding": self.encoding}).outputs()
        else:
            return (
                _get_factory()
                .create(
                    "CharsMapNormalization",
                    input_nodes,
                    {
                        "normalization_form": "identity",
                        "case_fold": True,
                        "remove_extra_whitespaces": False,
                    },
                )
                .outputs()
            )


@dataclass
class RegexNormalizationStep(NormalizationStep):
    regex_search_pattern: str
    replace_term: str
    global_replace: bool = True

    @classmethod
    def strip_accents_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"\p{Mn}", replace_term="")

    @classmethod
    def add_prefix_whitespace_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"^(\S)", replace_term=r" $1")
    
    @classmethod
    def replace_whitespace_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"\s", replace_term=" ", global_replace=True)

    @classmethod
    def handle_chinese_chars_regex(cls) -> "RegexNormalizationStep":
        return cls(
            regex_search_pattern=r"([\p{Han}])",
            replace_term=r" $1 ",
            global_replace=True,
        )

    @classmethod
    def add_prefix_whitespace_to_not_whitespace_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"^([^ ])", replace_term=r" $1")

    @classmethod
    def replace_spaces_metaspace(cls, replace_term=r"▁") -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r" ", replace_term=replace_term)

    @classmethod
    def prepend_regex(cls, string: str) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"(?:^)([\s\S])", replace_term=rf"{string}$1")

    @classmethod
    def prepend_with_check_regex(cls, string: str, check_string: str) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=rf"(^)([^{check_string}])", replace_term=rf"{string}$2")

    @classmethod
    def del_control_chars_regex(cls) -> "RegexNormalizationStep":
        return cls(
            regex_search_pattern=r"([\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\p{Cf}])",  # exclude \n\t\r
            replace_term="",
            global_replace=True,
        )

    @classmethod
    def strip_regex(cls, left: bool = True, right: bool = True) -> "RegexNormalizationStep":
        return cls(
            regex_search_pattern=r"^\s*" * left + "|" * (left and right) + r"\s*$" * right,
            replace_term="",
        )

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        input_nodes.extend(
            [
                *create_string_constant_node(self.regex_search_pattern),
                *create_string_constant_node(self.replace_term),
            ]
        )
        return (
            _get_factory().create("RegexNormalization", input_nodes, {"global_replace": self.global_replace}).outputs()
        )


@dataclass
class CharsmapStep(NormalizationStep):
    charsmap: Optional[bytes] = field(default=None, repr=False)
    normalization_form: Optional[str] = None
    add_dummy_prefix: bool = False
    remove_extra_whitespaces: bool = True
    escape_whitespaces: bool = False
    case_fold: bool = False
    nmt: bool = False

    def __add__(self, other: "CharsmapStep") -> "CharsmapStep":
        if self.charsmap is not None and other.charsmap is not None:
            raise ValueError("Cannot add two CharsmapStep instances with non-None charsmap attributes")
        if (
            self.normalization_form is not None
            and other.normalization_form is not None
            and self.normalization_form != "identity"
            and other.normalization_form != "identity"
            and self.normalization_form != other.normalization_form
        ):
            raise ValueError("Cannot add two CharsmapStep instances with different normalization_form attributes")

        return self.__class__(
            charsmap=self.charsmap or other.charsmap,
            normalization_form=self.normalization_form or other.normalization_form,
            add_dummy_prefix=self.add_dummy_prefix or other.add_dummy_prefix,
            remove_extra_whitespaces=self.remove_extra_whitespaces and other.remove_extra_whitespaces,
            escape_whitespaces=self.escape_whitespaces or other.escape_whitespaces,
            case_fold=self.case_fold or other.case_fold,
            nmt=self.nmt or other.nmt,
        )

    @classmethod
    def from_hf_step_json(cls, step_json: dict[str, Any]) -> "CharsmapStep":
        return cls(charsmap=base64.b64decode(step_json["precompiled_charsmap"]))

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        if self.charsmap is not None:
            input_nodes += make_constant_node(np.frombuffer(self.charsmap, dtype=np.uint8), dtype=Type.u8).outputs()
        return (
            _get_factory()
            .create(
                "CharsMapNormalization",
                input_nodes,
                {
                    "normalization_form": self.normalization_form or "",
                    "add_dummy_prefix": self.add_dummy_prefix,
                    "remove_extra_whitespaces": self.remove_extra_whitespaces,
                    "escape_whitespaces": self.escape_whitespaces,
                    "case_fold": self.case_fold,
                    "nmt": self.nmt,
                },
            )
            .outputs()
        )


@dataclass
class PreTokenizatinStep(BasePipelineStep):
    pass


@dataclass
class RegexSplitStep(PreTokenizatinStep):
    split_pattern: str
    invert: bool = False
    behaviour: str = "remove"
    max_splits: int = -1
    mergeable: bool = True

    def __post_init__(self):
        if self.max_splits < -1:
            raise ValueError(
                "RegexSplitStep max_splits attribute must be greater then `0` or equal to `-1`, "
                f"got `{self.max_splits}`"
            )

    def __add__(self, other: "RegexSplitStep") -> "RegexSplitStep":
        if not self.mergeable or not other.mergeable:
            raise ValueError("Cannot merge RegexSplitStep instances marked as non-mergeable")
        if self.invert != other.invert:
            raise ValueError("Cannot add two RegexSplitStep instances with different invert attributes")
        if self.behaviour != other.behaviour:
            raise ValueError("Cannot add two RegexSplitStep instances with different behaviour attributes")
        if self.behaviour != "remove" and self.behaviour != "isolate":
            raise ValueError(f'Only "remove" or "isolate" RegexSplit nodes can be merged, got {self.behaviour}')
        if self.max_splits != other.max_splits:
            raise ValueError("Cannot add two RegexSplitStep instances with different max_splits attributes")

        return self.__class__(
            split_pattern="|".join((self.split_pattern, other.split_pattern)),
            invert=self.invert,
            behaviour=self.behaviour,
            max_splits=self.max_splits,
        )

    @classmethod
    def split_by_chars(cls) -> "RegexSplitStep":
        return cls(split_pattern=".", invert=False, behaviour="isolate")

    @classmethod
    def bert_whitespace_splitter(cls) -> "RegexSplitStep":
        return cls(split_pattern=r"\s+", invert=False)

    @classmethod
    def bert_keep_delimeters_splitter(cls) -> "RegexSplitStep":
        """Generates a step with a standard BERT regex.

        The source:
        https://github.com/tensorflow/text/blob/4a098cd852c0b7ebee621e2d211c7f202dd679c2/tensorflow_text/python/ops/bert_tokenizer.py#L39
        """
        return cls(
            "|".join(
                [
                    r"|".join(
                        [
                            r"[!-/]",
                            r"[:-@]",
                            r"[\[-`]",
                            r"[{-~]",
                            r"[\p{P}]",
                        ],
                    ),
                    r"|".join(
                        [
                            r"[\x{4E00}-\x{9FFF}]",
                            r"[\x{3400}-\x{4DBF}]",
                            r"[\x{20000}-\x{2A6DF}]",
                            r"[\x{2A700}-\x{2B73F}]",
                            r"[\x{2B740}-\x{2B81F}]",
                            r"[\x{2B820}-\x{2CEAF}]",
                            r"[\x{F900}-\x{FAFF}]",
                            r"[\x{2F800}-\x{2FA1F}]",
                        ],
                    ),
                ],
            ),
            invert=False,
            behaviour="isolate",
        )

    @classmethod
    def bert_splitter(cls) -> list["RegexSplitStep"]:
        return [cls.bert_whitespace_splitter(), cls.bert_keep_delimeters_splitter()]

    @classmethod
    def whitespace_splitter(cls) -> "RegexSplitStep":
        return cls(r"\w+|[^\w\s]+", invert=True)

    @classmethod
    def metaspace_splitter(cls, metaspace=r"▁") -> "RegexSplitStep":
        return cls(metaspace, invert=False, behaviour="mergedwithnext")

    @classmethod
    def byte_level_splitter(cls, individual_digits: bool = False) -> "RegexSplitStep":
        if individual_digits:
            return cls(
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
                invert=False,
                behaviour="isolate",
            )
        return cls(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            invert=False,
            behaviour="isolate",
        )

    @classmethod
    def digits_splitter(cls, behaviour="isolate") -> "RegexSplitStep":
        return cls(
            r"\p{Nd}|\p{Nl}|\p{No}",
            invert=False,
            behaviour=behaviour,
        )

    @classmethod
    def punctuation_splitter(cls, behaviour="isolate") -> "RegexSplitStep":
        return cls(
            r"\p{P}",
            invert=False,
            behaviour=behaviour,
        )

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        input_nodes.extend(create_string_constant_node(self.split_pattern))
        return (
            _get_factory()
            .create(
                "RegexSplit",
                input_nodes,
                {
                    "behaviour": self.behaviour.lower(),
                    "invert": self.invert,
                    "max_splits": self.max_splits,
                },
            )
            .outputs()
        )


@dataclass
class WhitespaceSplitStep(PreTokenizatinStep):
    """Works like python `str.split`."""

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        return RegexSplitStep.whitespace_splitter().get_ov_subgraph(input_nodes)


@dataclass
class BytesToCharsStep(PreTokenizatinStep):
    """Maps chars to other chars for Byte-level BPE Tokenizer"""

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        return (
            _get_factory()
            .create(
                "BytesToChars",
                input_nodes,
            )
            .outputs()
        )


@dataclass
class TokenizationModelStep(BasePipelineStep):
    @staticmethod
    def get_vocab_as_list(vocab: dict[str, int]) -> list[str]:
        vocab_list = []
        for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            # Fill gaps up to token_id
            while len(vocab_list) < token_id:
                vocab_list.append("")

            if len(vocab_list) == token_id:  # place token at its id position
                vocab_list.append(token)
            else:
                # token_id < len(vocab) means duplicate/out-of-order id
                vocab_list[token_id] = token
        return vocab_list


@dataclass
class VocabEncoderStep(TokenizationModelStep):
    vocab: list[str] = field(repr=False)
    vocab_values: Optional[list[int]] = None
    default_value: int = -1

    def __post_init__(self) -> None:
        if self.vocab_values is None:
            self.vocab_values = list(range(len(self.vocab)))

    @classmethod
    def from_hf_json(cls, tokenizer_json: dict[str, Any]) -> "VocabEncoderStep":
        vocab = cls.get_vocab_as_list(tokenizer_json["model"]["vocab"])

        unk_token = tokenizer_json["model"].get("unk_token")
        unk_token_id = next((index for index, token in enumerate(vocab) if token == unk_token), -1)
        return cls(vocab, default_value=unk_token_id)

    def get_vocab_node_outputs(self) -> Optional[list[Output]]:
        return self.get_pipeline().vocab_node_outputs

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        pipeline = self.get_pipeline()
        pipeline.vocab_node_outputs = create_string_constant_node(self.vocab)

        ragged_dims, other_dims = [], input_nodes
        if len(input_nodes) > 4:
            ragged_dims, other_dims = input_nodes[:2], input_nodes[2:]
        other_dims.extend(
            (
                *pipeline.vocab_node_outputs,
                make_constant_node(np.array(self.vocab_values, dtype=np.int32), Type.i32),
                make_constant_node(self.default_value, Type.i32),  # default_value
            )
        )
        return ragged_dims + _get_factory().create("VocabEncoder", other_dims).outputs()


@dataclass
class TrieTokenizerStep(TokenizationModelStep):
    vocab: list[str] = field(repr=False)
    indices: list[int] = field(repr=False)

    def __post_init__(self):
        if len(self.vocab) != len(self.indices):
            raise UserInputError("Vocab and Indices must be the same length.")

        self.vocab, self.indices = self.fill_vocab(self.vocab, self.indices)

    @staticmethod
    def fill_vocab(vocab: list[str], indices: list[int]) -> tuple[list[str], list[int]]:
        max_idx = max(indices)
        new_indices = list(range(max_idx + 1))

        idx_to_token = dict(zip(indices, vocab))
        new_vocab = []
        for idx in new_indices:
            new_vocab.append(idx_to_token.get(idx, ""))

        return new_vocab, new_indices

    @classmethod
    def from_rwkv_vocab(cls, vocab_file_strings: Iterable[str]) -> TrieTokenizerStep:
        vocab = []
        indices = []
        for line in vocab_file_strings:
            idx = int(line.split(" ")[0])
            x = eval(line.split(" ", 1)[1].rsplit(" ", 1)[0])
            vocab.append(x)
            indices.append(idx)
        return cls(vocab, indices)

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        input_nodes.extend(
            (
                *create_string_constant_node(self.vocab),
                make_constant_node(np.array(self.indices, dtype=np.int32), Type.i32),
            )
        )
        return _get_factory().create("TrieTokenizer", input_nodes).outputs()


@dataclass
class WordPieceTokenizationStep(TokenizationModelStep):
    vocab: list[str] = field(repr=False)
    unk_token: str = "[UNK]"
    suffix_indicator: str = "##"
    max_bytes_per_word: int = 100
    unk_token_id: int = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.unk_token_id = self.vocab.index(self.unk_token)
        except ValueError:
            raise UserInputError(f"Cannot find unknown token '{self.unk_token}' in the vocab")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_hf_json(cls, tokenizer_json: dict[str, Any]) -> "WordPieceTokenizationStep":
        return cls(
            unk_token=tokenizer_json["model"]["unk_token"],
            suffix_indicator=tokenizer_json["model"]["continuing_subword_prefix"],
            vocab=cls.get_vocab_as_list(tokenizer_json["model"]["vocab"]),
        )

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        input_nodes.extend(
            (
                *create_string_constant_node(self.vocab),
                *as_node(self.unk_token_id).outputs(),
            )
        )
        return (
            _get_factory()
            .create(
                "WordpieceTokenizer",
                input_nodes,
                {
                    "suffix_indicator": self.suffix_indicator,
                    "max_bytes_per_word": self.max_bytes_per_word,
                },
            )
            .outputs()
        )


@dataclass
class BPETokenizationStep(TokenizationModelStep):
    vocab: Union[list[str], list[bytes]] = field(repr=False)
    merges: Union[list[str], list[tuple[bytes, bytes]]] = field(repr=False)
    unk_token: str = ""
    fuse_unk: bool = False
    suffix_indicator: str = ""
    end_suffix: str = ""
    byte_fallback: bool = False
    cache_capacity: int = MIN_CACHE_CAPACITY
    added_tokens: Optional[Union[dict[str, int], dict[bytes, int]]] = None

    def finalize(self) -> None:
        pipeline = self.get_pipeline()

        # remove BytesToChars transformation
        if pipeline.is_byte_level:
            self.vocab = [apply_unicode_to_bytes(token) for token in self.vocab]
            pipeline.vocab = self.vocab
            if self.merges_are_pairs:
                self.merges = [tuple(map(apply_unicode_to_bytes, merge)) for merge in self.merges]
            else:
                self.merges = [tuple(map(apply_unicode_to_bytes, merge.split(" "))) for merge in self.merges]

            # CharsToBytesStep might not be present in gguf-based tokenizers
            if any(isinstance(step, CharsToBytesStep) for step in pipeline.steps):
                chars_to_bytes_idx = next(
                    idx for idx, step in enumerate(pipeline.steps) if isinstance(step, CharsToBytesStep)
                )
                pipeline.steps.insert(chars_to_bytes_idx, FuseStep())
            pipeline.steps = [
                step for step in pipeline.steps if not isinstance(step, (BytesToCharsStep, CharsToBytesStep))
            ]

        if self.added_tokens is None:
            return

        if len(self.added_tokens.values()) > 0:
            size_diff = max(self.added_tokens.values()) - len(self.vocab) + 1
            if size_diff > 0:
                self.vocab.extend(type(self.vocab[0])() for _ in range(size_diff))

        for token, idx in self.added_tokens.items():
            if isinstance(self.vocab[0], bytes) and not isinstance(token, bytes):
                token = apply_unicode_to_bytes(token, return_corrupted_tokens=True)
            self.vocab[idx] = token

    @classmethod
    def from_hf_json(cls, tokenizer_json: dict[str, Any]) -> "BPETokenizationStep":
        vocab = cls.get_vocab_as_list(tokenizer_json["model"]["vocab"])

        added_tokens = {token["content"]: token["id"] for token in tokenizer_json["added_tokens"] if token["id"]}

        # TODO: CVS-150387 Implement suffix_indicator.
        if tokenizer_json["model"]["continuing_subword_prefix"]:
            raise NotImplementedError("continuing_subword_prefix/suffix_indicator is not implemented yet.")

        return cls(
            unk_token=tokenizer_json["model"]["unk_token"] or "",
            fuse_unk=tokenizer_json["model"]["fuse_unk"] or False,
            suffix_indicator=tokenizer_json["model"]["continuing_subword_prefix"] or "",
            end_suffix=tokenizer_json["model"]["end_of_word_suffix"] or "",
            vocab=vocab,
            merges=tokenizer_json["model"]["merges"],
            added_tokens=added_tokens,
            byte_fallback=tokenizer_json["model"]["byte_fallback"],
            cache_capacity=max(
                tokenizer_json["model"].get("cache_capacity", int(len(vocab) * VOCAB_SIZE_CACHE_PROPORTION)),
                MIN_CACHE_CAPACITY,
            ),
        )

    @classmethod
    def from_tiktoken_encoding(
        cls,
        encoding: "Encoding",  # noqa
        reference_vocab: Optional[dict[Union[str, bytes], int]] = None,
    ) -> "BPETokenizationStep":
        from .tiktoken_parser import generate_vocab_and_merges

        vocab, merges, added_tokens = generate_vocab_and_merges(encoding)
        added_tokens.update(dict(encoding._special_tokens.items()))

        if reference_vocab is not None:
            existing_indices = set(vocab.values())

            for ref_token, ref_idx in reference_vocab.items():
                if ref_idx in existing_indices:
                    continue

                # special tokens from reference vocab can be strings, not bytes
                if isinstance(ref_token, str):
                    ref_token = ref_token.encode()

                vocab[ref_token] = ref_idx

        return cls(
            unk_token="",
            fuse_unk=False,
            suffix_indicator="",
            end_suffix="",
            vocab=[token for token, idx in sorted(vocab.items(), key=lambda x: x[1])],
            merges=merges,
            added_tokens=added_tokens,
            cache_capacity=max(int(len(vocab) * VOCAB_SIZE_CACHE_PROPORTION), MIN_CACHE_CAPACITY),
        )

    @property
    def merges_are_pairs(self) -> bool:
        return self.merges and not isinstance(self.merges[0], str)

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        pipeline = self.get_pipeline()
        pipeline.vocab_node_outputs = create_string_constant_node(self.vocab)

        if self.added_tokens:
            special_tokens_outputs = create_string_constant_node(self.added_tokens)
        else:
            special_tokens_outputs = []

        # todo: check if this is still working after bytes-to-chars removal
        if special_tokens_outputs and pipeline.is_byte_level:
            special_tokens_outputs = pipeline.add_ragged_dimension(special_tokens_outputs)
            special_tokens_outputs = BytesToCharsStep().get_ov_subgraph(special_tokens_outputs)[-3:]

        input_nodes.extend(pipeline.vocab_node_outputs)
        if self.merges_are_pairs:
            left_merges, right_merges = zip(*self.merges)
            input_nodes.extend(
                (
                    *create_string_constant_node(left_merges),
                    *create_string_constant_node(right_merges),
                )
            )
        else:
            input_nodes.extend(create_string_constant_node(self.merges))

        if special_tokens_outputs:
            input_nodes.extend(
                (
                    *special_tokens_outputs,
                    *make_constant_node(np.array(list(self.added_tokens.values())), Type.i32).outputs(),
                )
            )

        return (
            _get_factory()
            .create(
                "BPETokenizer",
                input_nodes,
                {
                    "unk_token": self.unk_token,
                    "fuse_unk": self.fuse_unk,
                    "suffix_indicator": self.suffix_indicator,
                    "end_suffix": self.end_suffix,
                    "byte_fallback": self.byte_fallback,
                    "cache_capacity": self.cache_capacity,
                },
            )
            .outputs()
        )


@dataclass
class UnigramModelStep(TokenizationModelStep):
    vocab: list[Union[str, bytes]] = field(repr=False)
    vocab_logprobs: list[float] = field(repr=False)
    byte_fallback: bool = False
    unk_token_id: Optional[int] = None
    fuse_unk: bool = True
    min_score: float = float("inf")

    @classmethod
    def from_hf_json(cls, tokenizer_json: dict[str, Any]) -> "UnigramModelStep":
        vocab = tokenizer_json["model"]["vocab"]

        max_score = max(score for _, score in vocab)
        min_score = min(score for _, score in vocab)
        added_tokens = sorted((token["id"], token["content"]) for token in tokenizer_json.get("added_tokens", []))

        if added_tokens:
            max_added_token_id = added_tokens[-1][0]
            while max_added_token_id >= len(vocab):
                vocab.append(["", min_score])

        for added_token_id, token in added_tokens:
            # score for added tokens is (length * max_score_ - 0.1)
            vocab[added_token_id][0] = token
            vocab[added_token_id][1] = max(vocab[added_token_id][1], max_score * len(token) - 0.1)

        return cls(
            vocab=[token for token, _ in vocab],
            vocab_logprobs=[logprob for _, logprob in vocab],
            byte_fallback=tokenizer_json["model"]["byte_fallback"],
            unk_token_id=tokenizer_json["model"]["unk_id"],
        )

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        # Keep precision and not compress to f16 on ARM devices.
        const_vocab_logprobs_node =  make_constant_node(np.array(self.vocab_logprobs, dtype=np.float32), Type.f32)
        const_vocab_logprobs_node.get_rt_info()["precise_0"] = ""

        input_nodes.extend(
            (
                *create_string_constant_node(self.vocab),
               const_vocab_logprobs_node,
            )
        )
        return (
            _get_factory()
            .create(
                "UnigramTokenizer",
                input_nodes,
                {
                    "byte_fallback": self.byte_fallback,
                    "unk_token_id": self.unk_token_id,
                    "fuse_unk": self.fuse_unk,
                },
            )
            .outputs()
        )


@dataclass
class PostTokenizationStep(BasePipelineStep):
    pass


@dataclass
class TruncationStep(PostTokenizationStep):
    max_length: int
    truncate_right: bool = True
    axis: int = -1

    @classmethod
    def from_hf_json(
        cls, tokenizer_json: dict[str, Any], num_of_added_tokens: int = 0, max_length: int = -1
    ) -> "TruncationStep":
        if max_length == -1:
            max_length = min(
                tokenizer_json["truncation"]["max_length"] - num_of_added_tokens,
                2**31 - 1 - num_of_added_tokens,
            )
        else:
            max_length = min(max_length - num_of_added_tokens, 2**31 - 1 - num_of_added_tokens)
        return cls(
            max_length=max_length,
            truncate_right=tokenizer_json["truncation"]["direction"] == "Right",
        )

    @classmethod
    def from_hf_object(cls, tokenizer: Any, num_of_added_tokens: int = 0) -> "TruncationStep":
        max_length = min(
            tokenizer.model_max_length - num_of_added_tokens,
            2**31 - 1 - num_of_added_tokens,
        )
        return cls(
            max_length=max_length,
            truncate_right=tokenizer.truncation_side == "right",
        )

    @staticmethod
    def validate_inputs(input_nodes):
        if len(input_nodes) != 3:
            raise UserInputError("Only one input ragged tensor is supported as an input for TruncationStep")

    def get_ov_subgraph(self, input_nodes: list[Output]):
        # FIXME: Truncation side (truncate_right) is ignored
        # TODO: Check if axis is the right-most dimension
        self.validate_inputs(input_nodes)

        input_nodes.extend(make_constant_node(self.max_length, Type.i32).outputs())
        truncation_side = create_string_constant_node("right" if self.truncate_right else "left")
        truncation_mode = create_string_constant_node("longest_first")
        input_nodes.extend(truncation_side)
        input_nodes.extend(truncation_mode)

        return _get_factory().create("Truncate", input_nodes).outputs()


@dataclass
class SpecialTokenWithId:
    token: Optional[str] = None
    _token_id: Optional[int] = None

    def set_token_id(self, vocab: Optional[list[str]]) -> None:
        if self._token_id is None and vocab is not None and self.token in vocab:
            self._token_id = vocab.index(self.token)

    @property
    def token_id(self) -> Optional[int]:
        return self._token_id

    @token_id.setter
    def token_id(self, value: int) -> None:
        self._token_id = value


@dataclass
class TokenWithTypeId:
    token_type_id: Optional[int] = None


@dataclass
class AddToken(TokenWithTypeId, SpecialTokenWithId):
    enabled_by_default: bool = True
    pass


@dataclass
class Sequence(TokenWithTypeId):
    pass


@dataclass
class CombineSegmentsStep(PostTokenizationStep):
    inputs: list[TokenWithTypeId] = field(default_factory=list)
    segment_ids: Optional[list[int]] = None
    axis: int = -1
    add_special_tokens: bool = True

    def __post_init__(self):
        if self.segment_ids is not None:
            return

        segment_ids_tensor = [node.token_type_id for node in self.inputs]
        if any(segment is None for segment in segment_ids_tensor):
            segment_ids_tensor = [0] * len(self.inputs)

        self.segment_ids = segment_ids_tensor

    def finalize(self) -> None:
        pipeline = self.get_pipeline()
        self.set_tokens_ids(vocab=pipeline.vocab)

    def set_tokens_ids(self, vocab: Optional[list[int]]) -> None:
        for input_ in self.inputs:
            if isinstance(input_, AddToken) and input_.token_id is None:
                input_.set_token_id(vocab)

    @property
    def number_of_added_tokens(self) -> int:
        return sum(1 for input_ in self.inputs if (isinstance(input_, AddToken) and input_.enabled_by_default))

    @classmethod
    def from_hf_json_template_postprocessor(
        cls, post_processor_dict: dict[str, Any], number_of_inputs: int = 1, add_special_tokens: bool = True
    ) -> "CombineSegmentsStep":
        inputs: list[TokenWithTypeId] = []

        post_processor = post_processor_dict["single"]
        pair_post_processor = post_processor_dict["pair"]

        single_num_inputs = len(post_processor)
        pair_num_inputs = len(pair_post_processor)
        num_additional = pair_num_inputs - single_num_inputs
        start_from_idx = single_num_inputs - num_additional

        if number_of_inputs == 2 and num_additional != 2 and not start_from_idx >= 0:
            raise UserInputError("Only adding one additional pair for the second input is currently supported")

        is_two_inputs_supported = True

        # Assert that post_processor_dict for pair inputs is extended variant for single inputs
        for i in range(num_additional):
            pair_input = pair_post_processor[single_num_inputs + i]
            single_input = post_processor[start_from_idx + i]

            is_two_inputs_supported = pair_input.keys() == single_input.keys()
            if not is_two_inputs_supported:
                break
            for key in pair_input.keys():
                if key == "SpecialToken":
                    is_two_inputs_supported = pair_input[key]["id"] == single_input[key]["id"]

                    if not is_two_inputs_supported:
                        break

        if number_of_inputs == 2 and not is_two_inputs_supported:
            raise UserInputError(
                f"Two inputs not supported for this post-processors "
                f"single input post_processor {post_processor} "
                f"and pair input post_processor {pair_post_processor}"
            )

        for template_dict in post_processor:
            if "SpecialToken" in template_dict:
                step = AddToken(
                    token=template_dict["SpecialToken"]["id"],
                    token_type_id=template_dict["SpecialToken"]["type_id"],
                    enabled_by_default=add_special_tokens,
                )
                if special_tokens := post_processor_dict.get("special_tokens", False):
                    step.token_id = next(iter(special_tokens.get(step.token, {}).get("ids", [None])))
                inputs.append(step)
            elif "Sequence" in template_dict:
                inputs.append(Sequence(token_type_id=template_dict["Sequence"]["type_id"]))
        return cls(inputs, add_special_tokens=add_special_tokens)

    @classmethod
    def from_hf_json_bert_postprocessor(
        cls, post_processor_dict: dict[str, Any], number_of_inputs: int = 1, add_special_tokens: bool = True
    ) -> "CombineSegmentsStep":
        inputs: list[TokenWithTypeId] = []
        inputs.append(
            AddToken(token=post_processor_dict["cls"][0], token_type_id=0, enabled_by_default=add_special_tokens)
        )
        inputs[-1].token_id = post_processor_dict["cls"][1]
        inputs.append(Sequence(token_type_id=0))
        inputs.append(
            AddToken(token=post_processor_dict["sep"][0], token_type_id=0, enabled_by_default=add_special_tokens)
        )
        inputs[-1].token_id = post_processor_dict["sep"][1]

        return cls(inputs, add_special_tokens=add_special_tokens)

    @classmethod
    def from_hf_json_roberta_processor(
        cls, post_processor_dict: dict[str, Any], number_of_inputs: int = 1, add_special_tokens: bool = True
    ) -> "CombineSegmentsStep":
        inputs: list[TokenWithTypeId] = [Sequence(token_type_id=0)]

        inputs.insert(
            0,
            AddToken(
                token=post_processor_dict["cls"][0],
                _token_id=post_processor_dict["cls"][1],
                token_type_id=0,
                enabled_by_default=add_special_tokens,
            ),
        )
        inputs.append(
            AddToken(
                token=post_processor_dict["sep"][0],
                _token_id=post_processor_dict["sep"][1],
                token_type_id=0,
                enabled_by_default=add_special_tokens,
            )
        )
        return cls(inputs, add_special_tokens=add_special_tokens)

    def validate_inputs(self, input_nodes: list[Output]) -> None:
        number_of_sequence_inputs = sum(1 for input_ in self.inputs if isinstance(input_, Sequence))
        if number_of_sequence_inputs != len(input_nodes) / 3:
            raise UserInputError(
                f"Number of input nodes: {len(input_nodes)}, must be equal to {number_of_sequence_inputs}"
            )

    def get_ov_subgraph(self, input_nodes):
        self.validate_inputs(input_nodes)

        op_inputs = []
        input_nodes_iter = iter(input_nodes)

        segment_ids = []
        segment_index = 0
        for (key, token_id), group_iter in groupby(
            self.inputs, key=lambda input: (type(input), getattr(input, "token_id", None))
        ):
            if key is Sequence:
                for sequence in group_iter:
                    op_inputs.extend(islice(input_nodes_iter, 3))
                segment_ids.append(self.segment_ids[segment_index])
                segment_index += 1
            elif key is AddToken:
                ids = [node._token_id for node in group_iter]

                segment_ids.append(self.segment_ids[segment_index])
                segment_index += len(ids)

                op_inputs.extend(make_constant_node(0, Type.i32).outputs())
                # If we don't add special tokens then end is 0.
                op_inputs.extend(make_constant_node(len(ids) if self.add_special_tokens else 0, Type.i32).outputs())
                op_inputs.append(make_constant_node(np.array(ids), Type.i32).output(0))
            else:
                raise UserInputError(f"Unexpected node type in CombineSegments: {key}")

        op_inputs.append(make_constant_node(segment_ids, Type.i32).output(0))
        return _get_factory().create("CombineSegments", op_inputs).outputs()


@dataclass
class PaddingStep(PostTokenizationStep, SpecialTokenWithId):
    pad_right: bool = True
    token_type_id: Optional[int] = None
    max_length: int = -1
    axis: int = -1
    pad_to_max_length: bool = False

    @classmethod
    def from_hf_json(
        cls,
        tokenizer_json: dict[str, Any],
        pad_to_max_length: bool = False,
        max_length: int = -1,
        pad_right: bool = True,
    ) -> "PaddingStep":
        padding_dict = tokenizer_json["padding"]
        padding_strategy = padding_dict.get("strategy", {})
        if max_length == -1 and isinstance(padding_strategy, dict) and "Fixed" in padding_strategy:
            max_length = padding_strategy["Fixed"]

        return cls(
            token=padding_dict["pad_token"],
            _token_id=padding_dict["pad_id"],
            pad_right=pad_right,
            token_type_id=padding_dict["pad_type_id"],
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        )

    @staticmethod
    def validate_inputs(input_nodes: list[Output]) -> None:
        # Suppose input_nodes may have multiple tuples each with 3 tensors represented decomposed ragged tensors
        # We suppose that all ragged tensors represent the same structure and produce the mask only once
        if len(input_nodes) % 3 != 0 or len(input_nodes) < 3:
            raise UserInputError(
                f"Number of input nodes should be divisible by 3 and bigger or equal 3. Got {len(input_nodes)}"
            )

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        self.validate_inputs(input_nodes)

        outputs = []

        if not self.pad_to_max_length or self.max_length == -1 or self.max_length >= 2**31:
            # Calculate max_length as the maximum ragged length
            max_length = opset.reduce_max(
                opset.subtract(input_nodes[1], input_nodes[0]),
                make_constant_node(0, Type.i32),
            )
        else:
            max_length = make_constant_node(self.max_length, Type.i32)

        names = [TOKEN_IDS_INPUT_NAME, TOKEN_TYPE_IDS_INPUT_NAME][: len(input_nodes) // 3]
        for idx, name in enumerate(names):
            cur_outputs = (
                _get_factory()
                .create(
                    "RaggedToDense",
                    input_nodes[3 * idx : 3 * (idx + 1)]
                    + max_length.outputs()
                    + make_constant_node(self.token_id or 0, Type.i32).outputs(),
                    {
                        "pad_right": self.pad_right,
                        "pad_max_length": self.pad_to_max_length,
                    },
                )
                .outputs()
            )
            cur_outputs[0].tensor.add_names({name})

            outputs.append(cur_outputs[0])
            if idx == 0:
                mask = opset.convert(cur_outputs[1], "i32").output(
                    0
                )  # TODO: Change RaggedToDense to generate mask of any type

        outputs.append(mask)
        outputs[-1].add_names({ATTENTION_MASK_INPUT_NAME})

        return outputs


@dataclass
class DecodingStep(BasePipelineStep):
    pass


@dataclass
class VocabDecoderStep(DecodingStep):
    vocab: Optional[list[str]] = None
    skip_tokens: Optional[list[int]] = None
    do_skip_tokens: Optional[bool] = True

    def finalize(self) -> None:
        pipeline = self.get_pipeline()
        if pipeline is None and self.skip_tokens is None:
            self.skip_tokens = []
        elif self.skip_tokens is None:
            self.skip_tokens = pipeline.skip_tokens or []

    @staticmethod
    def add_special_tokens_to_vocab(vocab: list[str, bytes], added_tokens: dict[int, str]) -> list[str, bytes]:
        if not added_tokens:
            return vocab

        is_bytes = isinstance(vocab[0], bytes)
        for idx, token in added_tokens.items():
            if is_bytes:
                token = apply_unicode_to_bytes(token, return_corrupted_tokens=True)
            if idx < len(vocab):
                vocab[idx] = token
            else:
                while idx > len(vocab):
                    vocab.append(b"" if is_bytes else "")
                vocab.append(token)

        return vocab

    @classmethod
    def from_hf_json(
        cls,
        tokenizer_json: dict[str, Any],
        pipeline_vocab: Optional[list[str]],
        added_tokens: Optional[list[int]] = None,
        skip_tokens: Optional[list[int]] = None,
        do_skip_tokens: bool = True,
        is_byte_level: bool = False,
    ) -> "VocabDecoderStep":
        model_type = tokenizer_json["model"]["type"]

        if pipeline_vocab is not None and model_type == "WordLevel":
            vocab = [f" {token}" for token in pipeline_vocab]
        elif pipeline_vocab is not None and model_type == "WordPiece":
            vocab = [
                token if token in ".,!?" else token[2:] if token.startswith("##") else f" {token}"
                for token in pipeline_vocab
            ]
        elif pipeline_vocab is not None and is_byte_level:
            # corrupt tokens will be filtered from pipeline vocab, has to save them to match hf_tokenizer.decode output
            vocab = [apply_unicode_to_bytes(token, return_corrupted_tokens=True) for token in pipeline_vocab]
            vocab = cls.add_special_tokens_to_vocab(vocab, added_tokens)
        elif pipeline_vocab is not None and model_type == "Unigram":
            byte_fallback = tokenizer_json["model"]["byte_fallback"]
            vocab = [transform_unigram_token_to_bytes(token, byte_fallback) for token in pipeline_vocab]
        else:  # Use vocab node from pipeline
            vocab = None

        return cls(vocab, list(skip_tokens), do_skip_tokens)

    def get_vocab_node_outputs(self) -> Optional[list[Output]]:
        return self.get_pipeline().vocab_node_outputs if self.get_pipeline() is not None else None

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        if self.vocab is None:
            vocab_outputs = self.get_vocab_node_outputs()
        else:
            vocab_outputs = create_string_constant_node(self.vocab)
        input_nodes.extend(vocab_outputs)

        # Put constant with skip tokens even if do_skip_tokens=False, so that it can be switched on/off at runtime.
        # Slice through all skip tokens if flag is true, else slice to get an empty tensor.
        stop_const = op.Constant(Type.i32, Shape([1]), [np.iinfo(np.int32).max if self.do_skip_tokens else 0])

        zero_const = op.Constant(Type.i32, Shape([1]), [0])
        one_const = op.Constant(Type.i32, Shape([1]), [1])
        skip_tokens_const = op.Constant(Type.i32, Shape([len(self.skip_tokens)]), self.skip_tokens)
        sliced_skips = opset.slice(skip_tokens_const, zero_const, stop_const, one_const).outputs()
        input_nodes.extend(sliced_skips)

        return _get_factory().create("VocabDecoder", input_nodes).outputs()


@dataclass
class CharsToBytesStep(DecodingStep):
    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        return _get_factory().create("CharsToBytes", input_nodes, {}).outputs()


@dataclass
class FuseStep(DecodingStep):
    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        *input_nodes, chars_node = input_nodes
        return _get_factory().create("FuzeRagged", input_nodes, {}).outputs() + [chars_node]


@dataclass
class UTF8ValidateStep(DecodingStep):
    mode: UTF8ReplaceMode = field(default_factory=lambda: UTF8ReplaceMode.IGNORE)

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        replace_mode = True if self.mode == UTF8ReplaceMode.REPLACE else False
        return _get_factory().create("UTF8Validate", input_nodes, {"replace_mode": replace_mode}).outputs()


@dataclass
class ByteFallbackStep(DecodingStep):
    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        if len(input_nodes) == 5:
            ragged_dims, input_nodes = input_nodes[:2], input_nodes[2:]
        else:
            ragged_dims = []

        return ragged_dims + _get_factory().create("ByteFallback", input_nodes).outputs()


@dataclass
class RegexDecodingStep(DecodingStep):
    regex_search_pattern: str
    replace_term: str

    @classmethod
    def clean_up_tokenization_spaces(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=r"(?| ([\\.\\?\\!,])| ('[ms])| (') | ('[rv]e)| (n't))",
            replace_term=r"$1",
        )

    @classmethod
    def parse_replace_dict(cls, replace_dict: dict[str, Any]) -> "RegexDecodingStep":
        pattern = replace_dict.get("pattern", {}).get("String")
        content = replace_dict.get("content")
        if pattern is None or content is None:
            raise ValueError(f"Replace Decoding Op with this parameters: `{replace_dict}` does not support yet.")

        return cls(regex_search_pattern=pattern, replace_term=content)

    @classmethod
    def parse_strip_dict(cls, replace_dict: dict[str, Any]) -> "RegexDecodingStep":
        content = replace_dict.get("content")
        if content is None:
            raise ValueError(f"Replace Decoding Op with this parameters: `{replace_dict}` does not support yet.")

        return cls(regex_search_pattern=f"^{content}", replace_term="")

    @classmethod
    def rstrip_space(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=r" $",
            replace_term="",
        )

    @classmethod
    def strip_forward_space(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=r"^ ",
            replace_term="",
        )

    @classmethod
    def strip_forward_space_before_not_space(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=r"(^ )([^ ])",
            replace_term=r"$2",
        )

    @classmethod
    def replace_end_of_word_suffix(cls, suffix: str = "</w>") -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=suffix,
            replace_term=" ",
        )

    @classmethod
    def replace_continuing_subword_prefix(cls, prefix: str = "##") -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=prefix,
            replace_term="",
        )

    def get_ov_subgraph(self, input_nodes: list[Output]) -> list[Output]:
        if len(input_nodes) == 5:
            ragged_dims, input_nodes = input_nodes[:2], input_nodes[2:]
        else:
            ragged_dims = []

        input_nodes.extend(
            (
                *create_string_constant_node(self.regex_search_pattern),
                *create_string_constant_node(self.replace_term),
            )
        )
        return ragged_dims + _get_factory().create("RegexNormalization", input_nodes).outputs()

    @classmethod
    def replace_sp_spaces(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern="▁",
            replace_term=" ",
        )


@dataclass
class TokenizerPipeline:
    steps: list[BasePipelineStep] = field(default_factory=list)
    vocab: Optional[list[str]] = field(default=None, repr=False)
    added_tokens: Optional[list[str]] = field(default=None, repr=False)
    skip_tokens: Optional[list[int]] = field(default=None, repr=False)
    number_of_inputs: int = 1
    vocab_node_outputs: Optional[list[Output]] = field(default=None, repr=False)
    finalized: bool = False

    @property
    def is_byte_level(self) -> bool:
        return any(isinstance(step, BytesToCharsStep) for step in self.pre_tokenization_steps)

    def get_config(self) -> dict[str, dict[str, Any]]:
        return {type(step).__name__: step.get_config() for step in self.steps}

    @singledispatchmethod
    def add_steps(self, steps: Any) -> None:
        raise OVTypeError(f"Type {type(steps)} is not supported")

    @add_steps.register
    def _(self, steps: BasePipelineStep) -> None:
        self.steps.append(steps)
        steps.set_pipeline(self)

    @add_steps.register
    def _(self, steps: list) -> None:
        for step in steps:
            self.steps.append(step)
            step.set_pipeline(self)

    def __getitem__(self, item: int) -> BasePipelineStep:
        return self.steps[item]

    @staticmethod
    def replace_normalization_step(step: BasePipelineStep) -> BasePipelineStep:
        """
        Replaces the normalization steps with an equivalent Charsmap steps before merging.
        """
        if isinstance(step, CaseFoldStep) and step.encoding == "utf-8":
            return CharsmapStep(normalization_form="identity", case_fold=True, remove_extra_whitespaces=False)
        if isinstance(step, NormalizeUnicode):
            return CharsmapStep(normalization_form=step.normalization_form.lower(), remove_extra_whitespaces=False)

        return step

    def merge_normalization_steps(self) -> None:
        self.steps = [self.replace_normalization_step(step) for step in self.steps]

        charsmap_steps = [step for step in self.steps if isinstance(step, CharsmapStep)]
        if len(charsmap_steps) > 1:
            first_step_position = next(idx for idx, step in enumerate(self.steps) if isinstance(step, CharsmapStep))
            steps_without_charsmaps = [step for step in self.steps if not isinstance(step, CharsmapStep)]

            steps_without_charsmaps.insert(first_step_position, reduce(add, charsmap_steps))
            self.steps = steps_without_charsmaps

    def del_duplicated_split_steps(self) -> None:
        metaspace_split = next(
            (
                step
                for step in self.pre_tokenization_steps
                if (isinstance(step, RegexSplitStep) and step.split_pattern == "▁")
            ),
            None,
        )
        if not metaspace_split:
            return

        self.steps = [step for step in self.steps if not isinstance(step, WhitespaceSplitStep)]

    def merge_regex_split_steps(self) -> None:
        if not any(isinstance(step, RegexSplitStep) for step in self.pre_tokenization_steps):
            return

        first_step_position = next(idx for idx, step in enumerate(self.steps) if isinstance(step, RegexSplitStep))
        steps_without_pre_tokenization = [step for step in self.steps if not isinstance(step, RegexSplitStep)]

        old_regex_split_steps = [step for step in self.pre_tokenization_steps if isinstance(step, RegexSplitStep)]
        new_regex_split_steps = []
        while any(isinstance(step, RegexSplitStep) for step in old_regex_split_steps):
            step_idx, current_step = next(
                (idx, step) for idx, step in enumerate(old_regex_split_steps) if step is not None
            )
            old_regex_split_steps[step_idx] = None
            new_regex_split_steps.append(current_step)

            for idx, step in enumerate(old_regex_split_steps):
                if step is None:
                    continue

                try:
                    new_regex_split_steps[-1] = new_regex_split_steps[-1] + step
                    old_regex_split_steps[idx] = None
                except ValueError:
                    # If the steps can't be merged, we stop the inner loop
                    break

        steps_without_pre_tokenization[first_step_position:first_step_position] = new_regex_split_steps
        self.steps = steps_without_pre_tokenization

    def update_metaspace_step_with_special_tokens(self) -> None:
        """
        No metaspace insertion when input starts with special token.
        """
        if not self.is_metaspace_prepend_first:
            return
        special_tokens_split = next(
            (step for step in self.steps if isinstance(step, SpecialTokensSplit)),
            None,
        )
        if not special_tokens_split:
            return
        metaspace_step, special_tokens_split = self.steps[:2]

        metaspace_step.regex_search_pattern = r"(^)((?!{}| |$)|(?=[\r\n\t\f\v]))".format(
            "|".join(quote_meta(token.text) for token in special_tokens_split.special_tokens)
        )
        metaspace_step.global_replace = False

    def finalize(self) -> None:
        if self.finalized:
            return

        self.merge_normalization_steps()
        self.del_duplicated_split_steps()
        self.update_metaspace_step_with_special_tokens()

        for step in copy(self.steps):
            step.finalize()

        # merge after finalizing steps to make sure that BytesToCharsStep is removed
        self.merge_regex_split_steps()
        self.finalized = True

    @property
    def is_metaspace_prepend_first(self) -> bool:
        return isinstance(self.steps[0], RegexNormalizationStep)

    def get_tokenizer_ov_subgraph(self) -> Model:
        self.finalize()

        string_inputs = [op.Parameter(Type.string, PartialShape(["?"]))]

        processing_outputs = []
        for input_node in string_inputs:
            input_node = _get_opset_factory("opset15").create("StringTensorUnpack", input_node.outputs()).outputs()

            if self.is_metaspace_prepend_first:
                prepend_metaspace_step = self.steps.pop(0)
                input_node = prepend_metaspace_step.get_ov_subgraph(input_node)

            ragged = []
            if isinstance(self.steps[0], SpecialTokensSplit):
                input_node = self.add_ragged_dimension(input_node)
                input_node = self.steps[0].get_ov_subgraph(input_node)
                ragged, input_node = input_node[:2], input_node[2:]

            for step in self.normalization_steps:
                input_node = step.get_ov_subgraph(input_node)

            if not ragged:
                input_node = self.add_ragged_dimension(input_node)
            else:
                input_node = ragged + input_node

            for step in self.pre_tokenization_steps:
                input_node = step.get_ov_subgraph(input_node)

            for step in self.tokenization_steps:
                input_node = step.get_ov_subgraph(input_node[:-1])

            processing_outputs.extend(input_node)

        for step in self.post_tokenization_steps:
            processing_outputs = step.get_ov_subgraph(processing_outputs)

        model = Model(processing_outputs, string_inputs, name=TOKENIZER_NAME)
        return model

    @property
    def normalization_steps(self) -> list[NormalizationStep]:
        return [step for step in self.steps if isinstance(step, NormalizationStep)]

    @property
    def pre_tokenization_steps(self) -> list[PreTokenizatinStep]:
        return [step for step in self.steps if isinstance(step, PreTokenizatinStep)]

    @property
    def tokenization_steps(self) -> list[TokenizationModelStep]:
        return [step for step in self.steps if isinstance(step, TokenizationModelStep)]

    @property
    def post_tokenization_steps(self) -> list[PostTokenizationStep]:
        return [step for step in self.steps if isinstance(step, PostTokenizationStep)]

    @property
    def decoding_steps(self) -> list[DecodingStep]:
        return [step for step in self.steps if isinstance(step, DecodingStep)]

    @staticmethod
    def add_ragged_dimension(input_node: list[Output]) -> list[Output]:
        shape = opset.shape_of(input_node[0])
        batch_size = opset.gather(shape, as_node(0), as_node(0))
        ragged_begins = opset.range(as_node(0), batch_size, as_node(1), output_type="i32").outputs()
        ragged_ends = opset.range(
            as_node(1), opset.add(batch_size, make_constant_node(1, Type.i64)), as_node(1), output_type="i32"
        ).outputs()
        return ragged_begins + ragged_ends + input_node

    def create_decoding_pipeline(self, input_nodes: list[Output]) -> list[Output]:
        for step in self.decoding_steps:
            pipeline_step = step.get_ov_subgraph(input_nodes)
            input_nodes = pipeline_step

        return _get_opset_factory("opset15").create("StringTensorPack", input_nodes).outputs()

    def get_detokenizer_ov_subgraph(self) -> Model:
        self.finalize()

        if not any(isinstance(step, VocabDecoderStep) for step in self.decoding_steps):
            raise NotImplementedError("Detokenizer is not supported for this model yet!")

        input_node = op.Parameter(Type.i32, PartialShape(["?", "?"]))
        token_ids = input_node
        outputs = self.create_decoding_pipeline([token_ids])
        model = Model(outputs, [input_node], name=DETOKENIZER_NAME)
        model.output().tensor.add_names({STRING_OUTPUT_NAME})
        return model
