# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Unified CLI entry point for OpenVINO Tokenizers.

Usage:
    openvino_tokenizers convert            <hf_repo_id> [options]  – convert a HF tokenizer to OpenVINO
    openvino_tokenizers check              <hf_repo_id> [options]  – sanity-check a HF tokenizer
    openvino_tokenizers check_normalization <hf_repo_id> [options]  – test normalization steps only
    openvino_tokenizers diagnose           <hf_repo_id> [options]  – pipeline-level diagnostics
"""

import argparse


def main() -> None:
    from .check_normalization import _configure_parser as _cfg_check_norm
    from .check_normalization import run as _run_check_norm
    from .check_tokenizer import _configure_parser as _cfg_check
    from .check_tokenizer import run as _run_check
    from .convert_tokenizer import _configure_parser as _cfg_convert
    from .convert_tokenizer import run as _run_convert
    from .diagnose_tokenizer import _configure_parser as _cfg_diagnose
    from .diagnose_tokenizer import run as _run_diagnose

    parser = argparse.ArgumentParser(
        prog="openvino_tokenizers",
        description="OpenVINO Tokenizers CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    sub_convert = subparsers.add_parser(
        "convert",
        help="Convert a HuggingFace tokenizer to OpenVINO format.",
        description="Converts tokenizers from Huggingface Hub to OpenVINO Tokenizer model.",
    )
    _cfg_convert(sub_convert)
    sub_convert.set_defaults(func=_run_convert)

    sub_check = subparsers.add_parser(
        "check",
        help="Sanity-check a HuggingFace tokenizer through OpenVINO.",
        description=(
            "Quick sanity-check for a HuggingFace tokenizer:\n"
            "  [1] Load HF tokenizer\n"
            "  [2] Convert to OpenVINO (tokenizer + detokenizer)\n"
            "  [3] Compare encode/decode outputs on the standard test suite\n"
            "  [4] Run openvino_genai.Tokenizer encode/decode checks  (requires openvino_genai)\n"
            "  [5] Test batch padding and pair inputs  (requires openvino_genai)\n\n"
            "Exit code: 0 = all hard steps passed, 1 = any hard step failed.\n"
            "Step 5 warnings never affect the exit code."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _cfg_check(sub_check)
    sub_check.set_defaults(func=_run_check)

    sub_check_norm = subparsers.add_parser(
        "check_normalization",
        help="Test normalization steps of a HuggingFace tokenizer against their OV equivalents.",
        description=(
            "Normalization-step sanity-check for a HuggingFace tokenizer:\n"
            "  [1] Load HF tokenizer\n"
            "  [2] Print the parsed normalizer pipeline\n"
            "  [3] Compare HF and OV outputs per step + full pipeline\n\n"
            "Only fast tokenizers with a tokenizer.json are fully supported.\n"
            "Exit code: 0 = all steps passed, 1 = any mismatch or error."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _cfg_check_norm(sub_check_norm)
    sub_check_norm.set_defaults(func=_run_check_norm)

    sub_diagnose = subparsers.add_parser(
        "diagnose",
        help="Pipeline-level diagnostic for a HuggingFace tokenizer.",
        description=(
            "Pipeline-level diagnostic for a HuggingFace tokenizer:\n"
            "  [1] Load HF tokenizer\n"
            "  [2] Map tokenizer.json pipeline to OV steps\n"
            "  [3] Test normalization steps individually\n"
            "  [4] Test pre-tokenization\n"
            "  [5] Run full pipeline comparison\n\n"
            "Exit code: 0 = no issues, 1 = any mismatch or unsupported type."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _cfg_diagnose(sub_diagnose)
    sub_diagnose.set_defaults(func=_run_diagnose)

    args = parser.parse_args()
    args.func(args)
