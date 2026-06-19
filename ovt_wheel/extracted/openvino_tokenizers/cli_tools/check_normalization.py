#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Normalization-layer sanity-check CLI for a HuggingFace tokenizer.

Usage:
    openvino_tokenizers check_normalization <hf_repo_id> [options]

Steps performed:
  [1] Load the HF tokenizer and discover its normalizer step(s)
  [2] For each step, build an OV model + HF normalizer, compare on the
      standard test-string suite, then also compare the full stacked pipeline

Each normalizer entry in tokenizer.json is tested independently first so
mismatches can be attributed to a specific step.

Exit code: 0 = all steps passed, 1 = any mismatch or error.
"""

import argparse
import json
import sys
import textwrap
import traceback

# Re-use the standard test suite from check_tokenizer
from .check_tokenizer import (
    ALL_TEST_STRINGS,
    BOLD,
    GREEN,
    RED,
    RESET,
    YELLOW,
    _fail,
    _ok,
    _step,
    _truncate,
    _warn,
    step_load_tokenizer,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _build_hf_normalizer(step_dict: dict):
    """
    Wrap *step_dict* into a minimal ``tokenizers.Tokenizer`` and return the
    ``.normalizer`` object, which exposes ``.normalize_str()``.
    """
    from tokenizers import Tokenizer as _HFTokenizer

    minimal = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": step_dict,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        # WordLevel with an empty vocab is the lightest valid model;
        # unk_token must be a string (not null) to pass validation.
        "model": {"type": "WordLevel", "vocab": {}, "unk_token": "[UNK]"},
    }
    return _HFTokenizer.from_str(json.dumps(minimal)).normalizer


def _build_ov_normalizer(ov_steps: list):
    """
    Chain one or more ``NormalizationStep`` objects into a compiled OV model
    that accepts a batch of strings and returns a batch of strings.
    """
    from openvino import Core, Model, PartialShape, Type, op

    from openvino_tokenizers import _get_opset_factory

    input_node = op.Parameter(Type.string, PartialShape(["?"]))
    input_node.set_friendly_name("string_input")
    output = _get_opset_factory("opset15").create("StringTensorUnpack", input_node.outputs()).outputs()
    for step in ov_steps:
        output = step.get_ov_subgraph(output)
    output = _get_opset_factory("opset15").create("StringTensorPack", output).outputs()
    model = Model(output, [input_node], "normalizer")
    return Core().compile_model(model)


def _run_step_test(
    label: str,
    ov_compiled,
    hf_normalizer,
    test_strings: list[str],
) -> int:
    """
    Run every string in *test_strings* through both normalizers and compare.
    Prints a one-line ✓ / ✗ summary; on failure prints mismatching cases.
    Returns the number of mismatches.
    """
    mismatches: list[tuple[str, str, str]] = []

    for s in test_strings:
        try:
            expected = hf_normalizer.normalize_str(s)
        except Exception as exc:
            mismatches.append((s, f"<HF raised {type(exc).__name__}: {exc}>", ""))
            continue
        try:
            
            actual = str(ov_compiled([s])[0][0])
        except Exception as exc:
            mismatches.append((s, expected, f"<OV raised {type(exc).__name__}: {exc}>"))
            continue
        if expected != actual:
            mismatches.append((s, expected, actual))

    total = len(test_strings)
    tag = f"  {BOLD}{label}{RESET}"
    if not mismatches:
        print(f"{tag}: {GREEN}✓{RESET}  {total}/{total} matched")
    else:
        passed = total - len(mismatches)
        print(f"{tag}: {RED}✗{RESET}  {passed}/{total} matched — {len(mismatches)} failed", file=sys.stderr)
        for s, expected, actual in mismatches:
            print(f"\n    {YELLOW}Input:{RESET}    {_truncate(s)}", file=sys.stderr)
            print(textwrap.indent(f"Expected: {repr(expected)}", "    "), file=sys.stderr)
            print(textwrap.indent(f"Actual:   {repr(actual)}", "    "), file=sys.stderr)

    return len(mismatches)


def _get_normalizer_step_dicts(hf_tokenizer) -> list[dict] | None:
    """
    Return the list of individual normalizer step dicts from ``tokenizer.json``.
    Returns ``None`` if the tokenizer is not fast or has no ``tokenizer.json``.
    Returns ``[]`` if the JSON normalizer is ``null``.
    """
    import tempfile
    from pathlib import Path

    if not hf_tokenizer.is_fast:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_tokenizer.save_pretrained(tmpdir)
        tok_path = Path(tmpdir) / "tokenizer.json"
        if not tok_path.exists():
            return None
        with open(tok_path, encoding="utf-8") as f:
            tokenizer_json = json.load(f)

    normalizer = tokenizer_json.get("normalizer")
    if normalizer is None:
        return []

    if normalizer.get("type") == "Sequence":
        return normalizer["normalizers"]
    return [normalizer]


# ── main steps ────────────────────────────────────────────────────────────────

_LARGE_KEYS = {"precompiled_charsmap"}  # keys whose values are large blobs – print length only


def _format_step_dict(step_dict: dict) -> str:
    """Return a compact human-readable summary of a single normalizer step dict."""
    parts = []
    for k, v in step_dict.items():
        if k == "type":
            continue
        if k in _LARGE_KEYS or (isinstance(v, str) and len(v) > 60):
            parts.append(f"{k}=<{len(v)} chars>")
        elif isinstance(v, dict):
            inner = ", ".join(f"{ik}={repr(iv)}" for ik, iv in v.items())
            parts.append(f"{k}={{{inner}}}")
        else:
            parts.append(f"{k}={repr(v)}")
    return ", ".join(parts) if parts else ""


def _format_ov_step(ov_step) -> str:
    """Return a compact human-readable summary of a single OV NormalizationStep."""
    name = type(ov_step).__name__
    try:
        attrs = {k: v for k, v in vars(ov_step).items() if not k.startswith("_")}
    except Exception:
        attrs = {}
    if not attrs:
        return name
    parts = []
    for k, v in attrs.items():
        if isinstance(v, (bytes, str)) and len(v) > 60:
            parts.append(f"{k}=<{len(v)} chars>")
        else:
            parts.append(f"{k}={repr(v)}")
    return f"{name}({', '.join(parts)})" if parts else name


def step_print_pipeline(hf_tokenizer) -> list[dict] | None:
    """
    Step 2 – discover and print the normalizer pipeline (HF and OV).

    Returns the list of step dicts, or None if unavailable.
    """
    from openvino_tokenizers.hf_parser import TransformersTokenizerPipelineParser

    step_dicts = _get_normalizer_step_dicts(hf_tokenizer)

    if step_dicts is None:
        _warn("Not a fast tokenizer — normalizer pipeline cannot be extracted from tokenizer.json")
        return None

    if not step_dicts:
        _ok("No normalizer defined for this tokenizer")
        return []

    normalizers_map = TransformersTokenizerPipelineParser.normalizers_map

    print(f"\n  {len(step_dicts)} normalizer step(s):")
    for i, step_dict in enumerate(step_dicts, 1):
        step_type = step_dict.get("type", "<unknown>")
        hf_params = _format_step_dict(step_dict)
        hf_summary = f"{step_type}  {hf_params}" if hf_params else step_type

        # Translate to OV steps
        if step_type in normalizers_map:
            try:
                ov_steps = normalizers_map[step_type](step_dict)
                if not isinstance(ov_steps, list):
                    ov_steps = [ov_steps]
                ov_parts = [_format_ov_step(s) for s in ov_steps]
                ov_summary = None  # use ov_parts list
            except Exception as exc:
                ov_parts = None
                ov_summary = f"<error: {exc}>"
        else:
            ov_parts = None
            ov_summary = "<unsupported>"

        print(f"    {BOLD}{i}.{RESET} HF  {hf_summary}")
        if ov_parts is not None:
            for j, part in enumerate(ov_parts):
                prefix = "       OV  " if j == 0 else "           "
                print(f"{prefix}{part}")
        else:
            print(f"       OV  {ov_summary}")

    return step_dicts


def step_test_normalization(hf_tokenizer, step_dicts: list[dict], test_strings: list[str]) -> int:
    """
    Step 3: test each normalizer step individually, then test the full
    stacked pipeline.

    Returns the total number of mismatches across all steps (0 = all good).
    """
    from openvino_tokenizers.hf_parser import TransformersTokenizerPipelineParser

    if not step_dicts:
        _ok("No normalizer steps to test")
        return 0

    normalizers_map = TransformersTokenizerPipelineParser.normalizers_map

    total_failures = 0

    for step_dict in step_dicts:
        step_type = step_dict.get("type", "<unknown>")

        if step_type not in normalizers_map:
            _warn(f"  {BOLD}{step_type}{RESET}: unsupported normalizer type — skipping")
            continue

        # ── Build OV model ─────────────────────────────────────────────────
        try:
            ov_steps = normalizers_map[step_type](step_dict)
            if not isinstance(ov_steps, list):
                ov_steps = [ov_steps]
            ov_compiled = _build_ov_normalizer(ov_steps)
        except Exception as exc:
            _fail(f"  {BOLD}{step_type}{RESET}: failed to build OV model: {exc}")
            total_failures += 1
            continue

        # ── Build HF per-step normalizer ──────────────────────────────────
        try:
            hf_norm = _build_hf_normalizer(step_dict)
        except Exception as exc:
            _fail(f"  {BOLD}{step_type}{RESET}: failed to build HF normalizer: {exc}")
            total_failures += 1
            continue

        # ── Compare ───────────────────────────────────────────────────────
        failures = _run_step_test(step_type, ov_compiled, hf_norm, test_strings)
        total_failures += failures

    # ── Combined pipeline test ─────────────────────────────────────────────
    if len(step_dicts) > 1 and hasattr(hf_tokenizer, "backend_tokenizer"):
        hf_combined = hf_tokenizer.backend_tokenizer.normalizer
        if hf_combined is not None:
            # Build OV model with ALL steps chained
            all_ov_steps = []
            build_ok = True
            for step_dict in step_dicts:
                step_type = step_dict.get("type", "<unknown>")
                if step_type not in normalizers_map:
                    build_ok = False
                    break
                try:
                    steps = normalizers_map[step_type](step_dict)
                    all_ov_steps.extend(steps if isinstance(steps, list) else [steps])
                except Exception:
                    build_ok = False
                    break

            if build_ok and all_ov_steps:
                try:
                    ov_combined = _build_ov_normalizer(all_ov_steps)
                    failures = _run_step_test("combined pipeline", ov_combined, hf_combined, test_strings)
                    total_failures += failures
                except Exception as exc:
                    _warn(f"  Combined pipeline: could not run — {exc}")

    return total_failures


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openvino_tokenizers check_normalization",
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
    _configure_parser(parser)
    return parser


def run(args) -> None:
    total_steps = 3
    exit_code = 0
    n_strings = len(ALL_TEST_STRINGS)

    # ── Step 1 ────────────────────────────────────────────────────────────────
    _step(1, total_steps, f"Loading HF tokenizer '{args.repo_id}'")
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
    _step(2, total_steps, "Parsing normalizer pipeline")
    try:
        step_dicts = step_print_pipeline(hf_tokenizer)
    except Exception:
        _fail("Failed to parse normalizer pipeline:")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    if step_dicts is None:
        # Fast tokenizer not available — cannot extract or test
        print()
        sys.exit(1)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    _step(3, total_steps, f"Testing normalizer step(s) on {n_strings} strings")
    try:
        failures = step_test_normalization(hf_tokenizer, step_dicts, ALL_TEST_STRINGS)
        if failures:
            exit_code = 1
    except Exception:
        _fail("Normalization testing raised an unexpected exception:")
        traceback.print_exc(file=sys.stderr)
        exit_code = 1

    print()
    sys.exit(exit_code)


def check_normalization() -> None:
    run(build_parser().parse_args())
