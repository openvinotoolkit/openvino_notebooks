#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Quick sanity-check CLI for a single HuggingFace tokenizer.

Usage:
    check_tokenizer <hf_repo_id> [options]

Steps performed:
  [1] Load the HF tokenizer
  [2] Convert it to OpenVINO (tokenizer + detokenizer)
  [3] Run the full test-string suite and compare outputs
  [4] Run openvino_genai.Tokenizer encode/decode checks (only if openvino_genai is installed)
  [5] Run openvino_genai.Tokenizer padding + pair-input checks (only if openvino_genai is installed)
      — reported as errors for tokenizers-backend tokenizers, warnings otherwise

On success each step prints a single ✓ line.
On failure the step prints ✗ plus the relevant context (exception, failing
strings with expected/actual output diff).
"""

import argparse
import sys
import textwrap
import traceback
from typing import Optional

import numpy as np

from openvino_tokenizers.cli_tools.convert_tokenizer import check_positive_int

# ── test strings ─────────────────────────────────────────────────────────────

ENG_STRINGS = [
    "Eng... test, string?!",
    "Multiline\nstring!\nWow!",
    "A lot\t w!",
    "A lot\t\tof whitespaces!",
    "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n",
    "Eng, but with d1gits: 123; 0987654321, stop.0987654321 - eng, but with d1gits: 123",
    "USER: <image>\nWhat is in the image? ASSISTANT:",
    "What is OpenVINO?",
    (
        "If I have 100 million dollars, what kinds of projects should I invest to maximize "
        "my benefits in background of a growing number of artificial intelligence technologies?"
    ),
    (
        "Write an epic travel diary where an engineer, a poet, and a chef cross seven cities in seven nights, "
        "and in each city they must solve one unusual challenge: rebuild a clocktower using only recycled brass, "
        "compose a lullaby for a sleepless market, design a dinner menu for astronauts who miss home, map hidden "
        "canals beneath an old library, negotiate peace between rival street orchestras, restore a broken weather "
        "vane that predicts memories instead of storms, and finally present a public workshop explaining every "
        "decision, every tradeoff, every failed attempt, and every lesson learned, while also listing materials, "
        "budgets, timelines, contingency plans, and a final reflection on teamwork, creativity, responsibility, "
        "and how small practical choices can change the future of an entire neighborhood."
    ),
]
MULTILINGUAL_STRINGS = [
    "Тестовая строка!",
    "Testzeichenfolge?",
    "Tester, la chaîne...",
    "測試字符串",
    "سلسلة الاختبار",
    "מחרוזת בדיקה",
    "Сынақ жолы á",
    "رشته تست",
    "介绍下清华大学",
]
EMOJI_STRINGS = [
    "😀",
    "😁😁",
    "🤣🤣🤣😁😁😁😁",
    "🫠",
    "🤷‍♂️",
    "🤦🏼‍♂️",
]
MISC_STRINGS = [
    "",
    b"\x06".decode(),
    " ",
    " " * 10,
    " " * 256,
    "\n",
    " \t\n",
]

ALL_TEST_STRINGS = ENG_STRINGS + MULTILINGUAL_STRINGS + EMOJI_STRINGS + MISC_STRINGS

# Subset used for batch-padding and pair-input checks in step 5
PADDING_BATCH = [
    "What is OpenVINO?",
    "Multiline\nstring!\nWow!",
    "Eng, but with d1gits: 123; 0987654321, stop.0987654321 - eng, but with d1gits: 123",
    "Тестовая строка!",
    "😀",
]
PAIR_INPUTS = [
    ["What is OpenVINO?", "It is an inference toolkit."],
    ["Hello world", "Goodbye world, this is a longer second string."],
    ["Eng... test, string?!", "測試字符串"],
]

# ── helpers ───────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}", file=sys.stderr)


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n{BOLD}[{n}/{total}]{RESET} {msg}")


def _truncate(s: str, max_len: int = 80) -> str:
    s = repr(s)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _array_summary(arr) -> str:
    flat = arr.reshape(-1).tolist()
    if len(flat) <= 12:
        return str(flat)
    return str(flat[:6]) + " ... " + str(flat[-3:])


def _compare_outputs(hf_out: dict, ov_out, skip_missing: bool = False) -> list[str]:
    """
    Return a list of mismatch descriptions, empty list means all match.
    ov_out can be an OVDict (keyed by tensor name) or a plain dict.
    """
    issues = []
    for key, hf_val in hf_out.items():
        try:
            ov_val = ov_out[key]  # OVDict supports lookup by tensor name
        except (KeyError, RuntimeError):
            if skip_missing:
                continue
            issues.append(f"output key '{key}' missing from OV result")
            continue
        if ov_val.shape != hf_val.shape:
            issues.append(
                f"shape mismatch for '{key}': HF {hf_val.shape} vs OV {ov_val.shape}\n"
                f"    HF: {_array_summary(hf_val)}\n"
                f"    OV: {_array_summary(ov_val)}"
            )
        elif not np.all(ov_val == hf_val):
            issues.append(
                f"value mismatch for '{key}':\n"
                f"    HF: {_array_summary(hf_val)}\n"
                f"    OV: {_array_summary(ov_val)}"
            )
    return issues


# ── main steps ────────────────────────────────────────────────────────────────

def step_load_tokenizer(repo_id: str, use_fast: bool, trust_remote_code: bool, subfolder: Optional[str] = None):
    """Step 1 – load the HF tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
        subfolder=subfolder,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    cls_name = type(tokenizer).__name__
    _ok(f"Loaded  {BOLD}{repo_id}{RESET}  →  {cls_name}")
    return tokenizer


def step_convert(hf_tokenizer, with_detokenizer: bool, use_sentencepiece_backend: bool, max_length: int):
    """Step 2 – convert to OpenVINO."""
    from openvino import Core
    from openvino_tokenizers import convert_tokenizer

    result = convert_tokenizer(
        hf_tokenizer,
        with_detokenizer=with_detokenizer,
        use_sentencepiece_backend=use_sentencepiece_backend,
        max_length=max_length,
    )

    core = Core()
    if with_detokenizer:
        ov_tok_model, ov_detok_model = result
        ov_tok = core.compile_model(ov_tok_model)
        ov_detok = core.compile_model(ov_detok_model)
        _ok("Converted tokenizer + detokenizer")
        return ov_tok, ov_detok
    else:
        ov_tok = core.compile_model(result)
        _ok("Converted tokenizer")
        return ov_tok, None


def step_test_tokenizer(
    hf_tokenizer,
    ov_tokenizer,
    ov_detokenizer: Optional[object],
    add_special_tokens: bool,
    skip_special_tokens: bool,
    skip_missing_outputs: bool,
    max_length: int,
) -> int:
    """
    Step 3 – run the test suite.

    Returns the number of failing strings (0 = all good).
    """
    failures: list[tuple[str, list[str]]] = []

    for test_str in ALL_TEST_STRINGS:
        input_list = [test_str]

        # ── tokenizer comparison ─────────────────────────────────────────────
        hf_out = hf_tokenizer(
            input_list,
            return_tensors="np",
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        ov_out = ov_tokenizer(input_list)

        issues = _compare_outputs(dict(hf_out), ov_out, skip_missing=skip_missing_outputs)

        # ── detokenizer comparison ───────────────────────────────────────────
        if ov_detokenizer is not None:
            token_ids = hf_out["input_ids"]
            hf_decoded = hf_tokenizer.batch_decode(
                token_ids, skip_special_tokens=skip_special_tokens
            )
            ov_decoded = ov_detokenizer(token_ids.astype("int32"))["string_output"].tolist()
            if hf_decoded != ov_decoded:
                issues.append(
                    f"detokenizer mismatch:\n"
                    f"    HF: {hf_decoded}\n"
                    f"    OV: {ov_decoded}"
                )

        if issues:
            failures.append((test_str, issues))

    total = len(ALL_TEST_STRINGS)
    passed = total - len(failures)

    if not failures:
        _ok(f"All {total} strings matched")
    else:
        _fail(f"{passed}/{total} strings matched — {len(failures)} failed")
        for test_str, issues in failures:
            print(f"\n  {YELLOW}Input:{RESET} {_truncate(test_str)}", file=sys.stderr)
            for issue in issues:
                indented = textwrap.indent(issue, "    ")
                print(f"{RED}{indented}{RESET}", file=sys.stderr)

    return len(failures)


# ── openvino_genai step ───────────────────────────────────────────────────────

def _is_tokenizers_backend(hf_tokenizer) -> bool:
    """Return True when the HF tokenizer is built on the `tokenizers` library."""
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
        return isinstance(hf_tokenizer, TokenizersBackend)
    except ImportError:
        pass
    try:
        from transformers import PreTrainedTokenizerFast
        return isinstance(hf_tokenizer, PreTrainedTokenizerFast)
    except ImportError:
        return False


def _has_openvino_genai() -> bool:
    try:
        import openvino_genai  # noqa: F401
        return True
    except ImportError:
        return False


def step_test_genai(hf_tokenizer, saved_dir: str, skip_missing_outputs: bool, max_length: int) -> int:
    """
    Step 4 – load openvino_genai.Tokenizer from *saved_dir* and verify that
    encode / decode results match HuggingFace on the standard test strings.

    Mirrors test_encode, test_decode, and test_special_tokens from
    genai tokenizers tests.

    Returns the number of failing strings (0 = all good).
    """
    from openvino_genai import Tokenizer as GenAITokenizer

    genai_tok = GenAITokenizer(saved_dir)
    failures: list[tuple[str, list[str]]] = []

    for test_str in ALL_TEST_STRINGS:
        issues: list[str] = []

        # ── encode with / without special tokens (test_encode + test_special_tokens) ──
        for add_spec in (True, False):
            hf_ids = hf_tokenizer(
                [test_str], return_tensors="np", add_special_tokens=add_spec, truncation=True, max_length=max_length
            )["input_ids"][0]
            genai_ids = genai_tok.encode(test_str, add_special_tokens=add_spec).input_ids.data[0]
            if not np.array_equal(hf_ids, genai_ids):
                issues.append(
                    f"encode mismatch (add_special_tokens={add_spec}):\n"
                    f"    HF:    {_array_summary(hf_ids)}\n"
                    f"    GenAI: {_array_summary(genai_ids)}"
                )

        # ── decode with skip / keep special tokens (test_decode + test_special_tokens) ──
        hf_with_special = hf_tokenizer(
            [test_str], return_tensors="np", add_special_tokens=True, truncation=True, max_length=max_length
        )["input_ids"]
        for skip_spec in (True, False):
            hf_decoded = hf_tokenizer.decode(hf_with_special[0], skip_special_tokens=skip_spec)
            genai_decoded = genai_tok.decode(hf_with_special, skip_special_tokens=skip_spec)[0]
            if hf_decoded != genai_decoded:
                issues.append(
                    f"decode mismatch (skip_special_tokens={skip_spec}):\n"
                    f"    HF:    {repr(hf_decoded)}\n"
                    f"    GenAI: {repr(genai_decoded)}"
                )

        if issues:
            failures.append((test_str, issues))

    total = len(ALL_TEST_STRINGS)
    passed = total - len(failures)

    if not failures:
        _ok(f"All {total} strings matched")
    else:
        _fail(f"{passed}/{total} strings matched — {len(failures)} failed")
        for test_str, issues in failures:
            print(f"\n  {YELLOW}Input:{RESET} {_truncate(test_str)}", file=sys.stderr)
            for issue in issues:
                indented = textwrap.indent(issue, "    ")
                print(f"{RED}{indented}{RESET}", file=sys.stderr)

    return len(failures)


def step_test_genai_advanced(hf_tokenizer, saved_dir: str, strict: bool = False) -> int:
    """
    Step 5 – check batch padding and pair-input behaviour via
    openvino_genai.Tokenizer.

    When *strict* is False (default) issues are reported as warnings (⚠) and
    do NOT affect the exit code.  When *strict* is True (tokenizers-backend
    tokenizers) issues are reported as errors (✗) and DO affect the exit code.

    Mirrors test_padding and test_two_inputs_* from genai tokenizer tests.

    Returns the number of issue items (0 = all good).
    """
    from openvino_genai import Tokenizer as GenAITokenizer

    genai_tok = GenAITokenizer(saved_dir)
    warnings_list: list[tuple[str, str]] = []

    # ── Batch padding ─────────────────────────────────────────────────────────
    # Mirrors test_padding: varying add_special_tokens, max_length, padding_side.
    padding_cases = [
        # (label,                        add_spec, max_len, pad_to_max, pad_side)
        ("batch/longest/add_special",    True,     None,   None,       None),
        ("batch/longest/no_special",     False,    None,   None,       None),
        ("batch/max_length=32",          True,     32,     True,       None),
        ("batch/padding_side=left",      True,     None,   None,       "left"),
        ("batch/padding_side=right",     True,     None,   None,       "right"),
    ]
    for label, add_spec, max_len, pad_to_max, pad_side in padding_cases:
        hf_params: dict = dict(
            return_tensors="np",
            add_special_tokens=add_spec,
            padding="max_length" if pad_to_max else "longest",
            truncation=max_len is not None,
        )
        ov_params: dict = dict(add_special_tokens=add_spec)
        if max_len is not None:
            hf_params["max_length"] = max_len
            ov_params["max_length"] = max_len
            ov_params["pad_to_max_length"] = True
        if pad_side is not None:
            hf_params["padding_side"] = pad_side
            ov_params["padding_side"] = pad_side
        try:
            hf_res = hf_tokenizer(PADDING_BATCH, **hf_params)["input_ids"]
            genai_res = genai_tok.encode(PADDING_BATCH, **ov_params).input_ids.data
            if not np.array_equal(hf_res, genai_res):
                warnings_list.append((
                    label,
                    f"padding mismatch:\n"
                    f"    HF:    {_array_summary(hf_res)}\n"
                    f"    GenAI: {_array_summary(genai_res)}",
                ))
        except Exception as exc:
            warnings_list.append((label, f"raised {type(exc).__name__}: {exc}"))

    # ── Pair inputs ───────────────────────────────────────────────────────────
    # Mirrors test_two_inputs_string_list_of_lists: add_second_input is a GenAI
    # load-time property, no extra conversion required.
    try:
        genai_pair_tok = GenAITokenizer(saved_dir, {"add_second_input": True})

        for first, second in PAIR_INPUTS:
            label = f"pair/{first[:30]!r}"
            try:
                pair_hf = hf_tokenizer([[first, second]], return_tensors="np")["input_ids"]
                pair_genai = genai_pair_tok.encode([[first, second]]).input_ids.data
                if not np.array_equal(pair_hf, pair_genai):
                    warnings_list.append((
                        label,
                        f"pair encode mismatch:\n"
                        f"    HF:    {_array_summary(pair_hf)}\n"
                        f"    GenAI: {_array_summary(pair_genai)}",
                    ))
            except Exception as exc:
                warnings_list.append((label, f"raised {type(exc).__name__}: {exc}"))
    except Exception as exc:
        warnings_list.append((
            "pair_input_init",
            f"Could not load GenAI Tokenizer with add_second_input=True: {type(exc).__name__}: {exc}",
        ))

    if not warnings_list:
        _ok("No issues")
    elif strict:
        _fail(f"{len(warnings_list)} failure(s)")
        for label, detail in warnings_list:
            print(f"\n  {YELLOW}Case:{RESET} {label}", file=sys.stderr)
            indented = textwrap.indent(detail, "    ")
            print(f"{RED}{indented}{RESET}", file=sys.stderr)
    else:
        _warn(f"{len(warnings_list)} warning(s)")
        for label, detail in warnings_list:
            print(f"\n  {YELLOW}Case:{RESET} {label}", file=sys.stderr)
            indented = textwrap.indent(detail, "    ")
            print(f"{YELLOW}{indented}{RESET}", file=sys.stderr)

    return len(warnings_list)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("repo_id", help="HuggingFace model id or local path to a tokenizer directory.")
    parser.add_argument(
        "--subfolder",
        default=None,
        help="Tokenizer subfolder inside a HuggingFace repo or local model directory.",
    )
    parser.add_argument(
        "--use-fast-false",
        "--use_fast_false",
        action="store_true",
        default=False,
        help="Pass use_fast=False to AutoTokenizer.from_pretrained (legacy tokenizer).",
    )
    parser.add_argument(
        "--trust-remote-code",
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True to AutoTokenizer.from_pretrained.",
    )
    parser.add_argument(
        "--no-detokenizer",
        "--no_detokenizer",
        action="store_true",
        default=False,
        help="Skip detokenizer conversion and testing.",
    )
    parser.add_argument(
        "--use-sentencepiece-backend",
        "--use_sentencepiece_backend",
        action="store_true",
        default=False,
        help="Use the SentencePiece backend during conversion.",
    )
    parser.add_argument(
        "--no-special-tokens",
        "--no_special_tokens",
        action="store_true",
        default=False,
        help="Encode test strings without special tokens.",
    )
    parser.add_argument(
        "--no-skip-special-tokens",
        "--no_skip_special_tokens",
        dest="skip_special_tokens",
        action="store_false",
        default=True,
        help="Decode test tokens with skip_special_tokens=False (default is True, matching the OV detokenizer default).",
    )
    parser.add_argument(
        "--skip-missing-outputs",
        "--skip_missing_outputs",
        action="store_true",
        default=False,
        help="Ignore HF outputs that are absent in the OV result (e.g. token_type_ids).",
    )
    
    parser.add_argument(
        "--max-length",
        "--max_length",
        type=check_positive_int,
        default=None,
        help="Set max_length for tokenizer conversion and HF truncation checks (default: None).",
    )


def run(args) -> None:
    has_genai = _has_openvino_genai()
    total_steps = 5 if has_genai else 3
    exit_code = 0

    # ── Step 1 ────────────────────────────────────────────────────────────────
    _step(1, total_steps, f"Loading HF tokenizer  '{args.repo_id}'")
    try:
        hf_tokenizer = step_load_tokenizer(
            repo_id=args.repo_id,
            use_fast=not args.use_fast_false,
            trust_remote_code=args.trust_remote_code,
            subfolder=args.subfolder,
        )
    except Exception:
        _fail("Failed to load tokenizer:")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    _step(2, total_steps, "Converting to OpenVINO")
    saved_dir = None
    try:
        ov_tokenizer, ov_detokenizer = step_convert(
            hf_tokenizer=hf_tokenizer,
            with_detokenizer=not args.no_detokenizer,
            use_sentencepiece_backend=args.use_sentencepiece_backend,
            max_length=args.max_length,
        )
        if has_genai:
            import tempfile
            from openvino import save_model
            from openvino_tokenizers import convert_tokenizer
            saved_dir = tempfile.mkdtemp(prefix="check_tokenizer_")
            ov_tok_model, ov_detok_model = convert_tokenizer(
                hf_tokenizer,
                with_detokenizer=True,
                use_sentencepiece_backend=args.use_sentencepiece_backend,
                max_length=args.max_length,
            )
            save_model(ov_tok_model, f"{saved_dir}/openvino_tokenizer.xml")
            save_model(ov_detok_model, f"{saved_dir}/openvino_detokenizer.xml")
    except Exception:
        _fail("Conversion failed:")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    n_strings = len(ALL_TEST_STRINGS)
    _step(3, total_steps, f"Testing against {n_strings} strings")
    try:
        n_failures = step_test_tokenizer(
            hf_tokenizer=hf_tokenizer,
            ov_tokenizer=ov_tokenizer,
            ov_detokenizer=ov_detokenizer,
            add_special_tokens=not args.no_special_tokens,
            skip_special_tokens=args.skip_special_tokens,  # True by default
            skip_missing_outputs=args.skip_missing_outputs,
            max_length=args.max_length,
        )
        if n_failures:
            exit_code = 1
    except Exception:
        _fail("Testing raised an unexpected exception:")
        traceback.print_exc(file=sys.stderr)
        exit_code = 1

    # ── Step 4 (optional) ─────────────────────────────────────────────────────
    if has_genai and saved_dir is not None:
        _step(4, total_steps, "Testing via openvino_genai.Tokenizer (encode + decode + special tokens)")
        try:
            n_genai_failures = step_test_genai(
                hf_tokenizer=hf_tokenizer,
                saved_dir=saved_dir,
                skip_missing_outputs=args.skip_missing_outputs,
                max_length=args.max_length,
            )
            if n_genai_failures:
                exit_code = 1
        except Exception:
            _fail("GenAI testing raised an unexpected exception:")
            traceback.print_exc(file=sys.stderr)
            exit_code = 1

    # ── Step 5 (optional, warnings or errors depending on backend) ────────
    if has_genai and saved_dir is not None:
        strict_advanced = _is_tokenizers_backend(hf_tokenizer)
        label = "errors" if strict_advanced else "warnings only"
        _step(5, total_steps, f"Testing openvino_genai.Tokenizer padding + pair inputs ({label})")
        try:
            n_advanced = step_test_genai_advanced(
                hf_tokenizer=hf_tokenizer,
                saved_dir=saved_dir,
                strict=strict_advanced,
            )
            if strict_advanced and n_advanced:
                exit_code = 1
        except Exception:
            if strict_advanced:
                _fail("Advanced GenAI testing raised an unexpected exception:")
                exit_code = 1
            else:
                _warn("Advanced GenAI testing raised an unexpected exception:")
            traceback.print_exc(file=sys.stderr)
    elif not has_genai:
        print("\n  (skipping steps 4-5 — openvino_genai not installed)")

    if saved_dir is not None:
        import shutil
        shutil.rmtree(saved_dir, ignore_errors=True)

    print()
    sys.exit(exit_code)
