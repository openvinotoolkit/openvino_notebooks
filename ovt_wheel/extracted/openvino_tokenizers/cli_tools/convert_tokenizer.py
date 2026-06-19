# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import Action, ArgumentError, ArgumentParser
from pathlib import Path

from openvino import Type, save_model

from openvino_tokenizers import convert_tokenizer
from openvino_tokenizers.constants import UTF8ReplaceMode


class StringToTypeAction(Action):
    string_to_type_dict = {
        "i32": Type.i32,
        "i64": Type.i64,
    }

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, self.string_to_type_dict[values])


def check_positive_int(value: str) -> int:
    int_value = int(value)
    if int_value <= 0:
        raise ArgumentError(f"Value must be positive integer, got: {value}")
    return int_value


class TrueOrPositiveIntAction(Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if values.isnumeric():
            values = int(values)
            if values > 0:
                setattr(namespace, self.dest, values)
                return

        if isinstance(values, str):
            if values.lower() == "true":
                setattr(namespace, self.dest, True)
                return
        raise ValueError(f'Value for {self.dest} must be positive integer or "True", got: {values}')


def _configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument(
        "name",
        type=str,
        help=(
            "The model id of a tokenizer hosted inside a model repo on huggingface.co "
            "or a path to a saved Huggingface tokenizer directory."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(),
        required=False,
        help="Output directory",
    )
    parser.add_argument(
        "--with-detokenizer",
        "--with_detokenizer",
        required=False,
        action="store_true",
        help="Add a detokenizer model to the output",
    )
    parser.add_argument(
        "--subfolder",
        required=False,
        type=str,
        default="",
        help=(
            "Specify in case the tokenizer files are located inside a subfolder of the model repo on huggingface.co. "
            "Example: `convert_tokenizer SimianLuo/LCM_Dreamshaper_v7 --subfolder tokenizer`"
        ),
    )
    parser.add_argument(
        "--not-add-special-tokens",
        "--not_add_special_tokens",
        required=False,
        action="store_false",
        help=(
            "Tokenizer won't add special tokens during tokenization, similar to "
            "huggingface_tokenizer.encode(texts, add_special_tokens=False). Not affects tiktoken-based tokenizers."
        ),
    )
    parser.add_argument(
        "--left_padding",
        "--left-padding",
        required=False,
        action="store_true",
        help="Tokenizer will add padding tokens to the left side. Not supported for Sentencepiece-based tokenizers.",
    )
    parser.add_argument(
        "--number_of_inputs",
        "--number-of-inputs",
        required=False,
        default=1,
        action=TrueOrPositiveIntAction,
        help=("The number of inputs for the model. Default is 1."),
    )
    parser.add_argument(
        "--max_padding",
        "--max-padding",
        required=False,
        action=TrueOrPositiveIntAction,
        help=(
            "Tokenizer will add padding tokens to max input size, "
            'similar to huggingface_tokenizer(text, padding="max_length"). '
            "You can pass a positive integer that can be used as max length parameter "
            'or "True" to use `huggingface_tokenizer.model_max_length` value (if set). '
            "Not supported for Sentencepiece-based tokenizers."
        ),
    )
    parser.add_argument(
        "--max_length",
        "--max-length",
        required=False,
        type=check_positive_int,
        help=(
            "Set max_length to the tokenizer for truncation operation. "
            "Tokenizer won't produce output longer than max_length. "
            "The value will be replaced by the max_padding option if set."
        ),
    )
    skip_special_group = parser.add_mutually_exclusive_group()
    skip_special_group.add_argument(
        "--not-skip-special-tokens",
        "--not_skip_special_tokens",
        required=False,
        action="store_false",
        help=(
            "Produce detokenizer that won't skip special tokens during decoding, similar to "
            "huggingface_tokenizer.decode(token_ids, skip_special_tokens=False)."
            "Not compatible with --skip-special-tokens."
        ),
    )
    skip_special_group.add_argument(
        "--skip-special-tokens",
        "--skip_special_tokens",
        required=False,
        action="store_true",
        default=True,
        help=(
            "Produce detokenizer that will skip special tokens during decoding, similar to "
            "huggingface_tokenizer.decode(token_ids, skip_special_tokens=True). "
            "This is the default behaviour, so adding this option has no effect, for backward compatibility only. "
            "Not compatible with --not-skip-special-tokens."
        ),
    )
    parser.add_argument(
        "--clean-up-tokenization-spaces",
        "--clean_up_tokenization_spaces",
        required=False,
        type=lambda x: {"True": True, "False": False}.get(x),
        default=None,
        choices=[True, False],
        help=(
            "Produce detokenizer that will clean up spaces before punctuation during decoding, similar to "
            "huggingface_tokenizer.decode(token_ids, clean_up_tokenization_spaces=True). This option is often set "
            "to False for code generation models. If the option is not set, a value from "
            "huggingface_tokenizer.clean_up_tokenization_spaces will be used."
        ),
    )
    parser.add_argument(
        "--use-fast-false",
        "--use_fast_false",
        required=False,
        action="store_false",
        help=(
            "Pass `use_fast=False` to `AutoTokenizer.from_pretrained`. It will initialize legacy HuggingFace "
            "tokenizer and then converts it to OpenVINO. Might result in slightly different tokenizer. "
            "See models with _slow suffix https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/"
            "custom_operations/user_ie_extensions/tokenizer/python#output-match-by-model to check the potential "
            "difference between original and OpenVINO tokenizers."
        ),
    )
    parser.add_argument(
        "--use-sentencepiece-backend",
        "--use_sentencepiece_backend",
        required=False,
        action="store_true",
        help=(
            "Use Sentencepiece library as a backend for tokenizer operation. "
            "The repository should contain Sentencepiece `.model` file. "
            "Unigram models supported by Sentencepiece backend only."
        ),
    )
    parser.add_argument(
        "--handle-special-tokens-with-re",
        "--handle_special_tokens_with_re",
        required=False,
        action="store_true",
        help=(
            "Use a regex to handle special tokens for tokenizers with Sentencepiece backed. "
            "Use this option if the converted tokenizer doesn't recognize special tokens during tokenization."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        "--trust_remote_code",
        required=False,
        action="store_true",
        help=(
            "Pass `trust_remote_code=True` to `AutoTokenizer.from_pretrained`. It will "
            "execute code present on the Huggingface Hub on your machine!"
        ),
    )
    parser.add_argument(
        "--tokenizer-output-type",
        "--tokenizer_output_type",
        required=False,
        action=StringToTypeAction,
        default=Type.i64,
        choices=["i32", "i64"],
        help="Type of the output tensors for tokenizer.",
    )
    parser.add_argument(
        "--detokenizer-input-type",
        "--detokenizer_input_type",
        required=False,
        action=StringToTypeAction,
        default=Type.i64,
        choices=["i32", "i64"],
        help="Type of the input tensor for detokenizer.",
    )
    parser.add_argument(
        "--streaming-detokenizer",
        "--streaming_detokenizer",
        required=False,
        action="store_true",
        help=(
            "[Experimental] Modify SentencePiece based detokenizer to keep spaces leading space. "
            "Can be used to stream a model output without TextStreamer buffer."
        ),
    )
    parser.add_argument(
        "--utf8_replace_mode",
        choices=list(UTF8ReplaceMode),
        type=UTF8ReplaceMode,  # enum with 'ignore', 'replace' values.
        default=UTF8ReplaceMode.REPLACE,
        required=False,
        help=(
            "If specified then resulting strings during decoding are checked if sequence of bytes is a valid UTF-8 sequence. "
            f"If mode is '{UTF8ReplaceMode.DISABLE}' then UTF8 validation is not performed at all. "
            f"Two other regimes are identical to python decode method error handling parameter. "
            f"If mode is '{UTF8ReplaceMode.REPLACE}' then invalid characters are replaced with \N{REPLACEMENT CHARACTER}. "
            f"if mode is '{UTF8ReplaceMode.IGNORE}' then invalid character are skipped and instead of them empty substring is added."
        ),
    )

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="openvino_tokenizers convert",
        description="Converts tokenizers from Huggingface Hub to OpenVINO Tokenizer model.",
    )
    _configure_parser(parser)
    return parser


def run(args) -> None:
    try:
        from transformers import AutoTokenizer
    except (ImportError, ModuleNotFoundError):
        raise EnvironmentError(
            "No transformers library in the environment. Install required dependencies with one of two options:\n"
            "1. pip install openvino-tokenizers[transformers]\n"
            "2. pip install transformers[sentencepiece] tiktoken\n"
        )

    tokenizer_init_kwargs = {
        "subfolder": args.subfolder,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.left_padding:
        tokenizer_init_kwargs["padding_side"] = "left"

    print("Loading Huggingface Tokenizer...")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.name, **tokenizer_init_kwargs)

    if isinstance(args.max_padding, int) and args.max_padding is not True:
        print(f"Set max_length to: {args.max_padding}")
        hf_tokenizer.model_max_length = args.max_padding
    elif args.max_length:
        print(f"Set max_length to: {args.max_length}")
        hf_tokenizer.model_max_length = args.max_length

    print("Converting Huggingface Tokenizer to OpenVINO...")
    converted = convert_tokenizer(
        hf_tokenizer,
        with_detokenizer=args.with_detokenizer,
        skip_special_tokens=args.not_skip_special_tokens,
        add_special_tokens=args.not_add_special_tokens,
        clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
        tokenizer_output_type=args.tokenizer_output_type,
        detokenizer_input_type=args.detokenizer_input_type,
        streaming_detokenizer=args.streaming_detokenizer,
        use_max_padding=args.max_padding is not None,
        handle_special_tokens_with_re=args.handle_special_tokens_with_re,
        use_sentencepiece_backend=args.use_sentencepiece_backend,
        utf8_replace_mode=args.utf8_replace_mode,
        max_length=args.max_length,
        number_of_inputs=args.number_of_inputs,
    )
    if not isinstance(converted, tuple):
        converted = (converted,)

    for converted_model, name in zip(converted, ("tokenizer", "detokenizer")):
        save_path = args.output / f"openvino_{name}.xml"
        save_model(converted_model, save_path)
        print(f"Saved OpenVINO {name.capitalize()}: {save_path}, {save_path.with_suffix('.bin')}")


def convert_hf_tokenizer() -> None:
    run(get_parser().parse_args())
