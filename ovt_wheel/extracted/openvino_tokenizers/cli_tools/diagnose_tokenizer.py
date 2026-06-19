#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Pipeline-level diagnostic CLI for a tokenizers.

Usage:
    openvino_tokenizers diagnose <hf_repo_id> [options]

Steps performed:
  [1] Load the HF tokenizer
  [2] Extract tokenizer.json pipeline and map to OV steps
  [3] Run normalization diagnostics (per-step + combined)
  [4] Run pre-tokenization diagnostics
  [5] Run full pipeline comparison to locate first point of divergence

Exit code: 0 = all stages matched, 1 = any mismatch or unsupported type.
"""

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

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

# ── formatting helpers ────────────────────────────────────────────────────────

_LARGE_KEYS = {"precompiled_charsmap", "vocab", "merges", "scores"}


def _format_hf_step(step_dict: dict) -> str:
    """Compact human-readable summary of an HF step dict."""
    step_type = step_dict.get("type", "<unknown>")
    parts = []
    for k, v in step_dict.items():
        if k == "type":
            continue
        if k in _LARGE_KEYS or (isinstance(v, (str, list)) and len(v) > 60):
            parts.append(f"{k}=<{len(v)} items>")
        elif isinstance(v, dict):
            inner = ", ".join(f"{ik}={repr(iv)}" for ik, iv in v.items())
            parts.append(f"{k}={{{inner}}}")
        else:
            parts.append(f"{k}={repr(v)}")
    params = ", ".join(parts)
    return f"{step_type}({params})" if params else step_type


def _format_ov_step(ov_step) -> str:
    """Compact human-readable summary of an OV pipeline step object."""
    name = type(ov_step).__name__
    try:
        attrs = {k: v for k, v in vars(ov_step).items() if not k.startswith("_")}
    except Exception:
        attrs = {}
    if not attrs:
        return name
    parts = []
    for k, v in attrs.items():
        if isinstance(v, (bytes, str, list)) and len(v) > 60:
            parts.append(f"{k}=<{len(v)} items>")
        else:
            parts.append(f"{k}={repr(v)}")
    return f"{name}({', '.join(parts)})" if parts else name


# ── tokenizer.json extraction ────────────────────────────────────────────────


def _load_tokenizer_json(hf_tokenizer) -> dict | None:
    """Save the HF tokenizer to a temp dir and load its tokenizer.json."""
    if not hf_tokenizer.is_fast:
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_tokenizer.save_pretrained(tmpdir)
        tok_path = Path(tmpdir) / "tokenizer.json"
        if not tok_path.exists():
            return None
        with open(tok_path, encoding="utf-8") as f:
            return json.load(f)


def _get_section_steps(section_json) -> list[dict]:
    """
    Flatten a tokenizer.json section into a list of step dicts.
    Handles both single-step and Sequence wrappers.
    """
    if section_json is None:
        return []
    if section_json.get("type") == "Sequence":
        key = None
        for candidate in ("normalizers", "pretokenizers", "processors", "decoders"):
            if candidate in section_json:
                key = candidate
                break
        if key:
            return section_json[key]
    return [section_json]


# ── pipeline mapping ─────────────────────────────────────────────────────────


_SECTION_NAMES = ["normalizer", "pre_tokenizer", "model", "post_processor", "decoder"]


def step_print_pipeline_map(tokenizer_json: dict, parser_class) -> dict:
    """
    Step 2 — print the HF → OV mapping for every pipeline section.

    Returns a dict of {section_name: {"hf_steps": [...], "supported": bool, "unsupported_types": [...]}}
    """
    section_info = {}

    for section_name in _SECTION_NAMES:
        section_json = tokenizer_json.get(section_name)
        hf_steps = _get_section_steps(section_json)

        if not hf_steps:
            section_info[section_name] = {"hf_steps": [], "supported": True, "unsupported_types": []}
            print(f"\n  {BOLD}{section_name}{RESET}: (none)")
            continue

        # Determine which map to use for support checking
        type_map = _get_type_map_for_section(section_name, parser_class)
        unsupported = []

        print(f"\n  {BOLD}{section_name}{RESET}:")
        for i, step_dict in enumerate(hf_steps, 1):
            step_type = step_dict.get("type", "<unknown>")
            hf_summary = _format_hf_step(step_dict)

            if type_map is not None and step_type not in type_map:
                print(f"    {i}. HF  {hf_summary}")
                print(f"       OV  {RED}⚠ UNSUPPORTED{RESET}")
                unsupported.append(step_type)
            else:
                ov_repr = _try_map_step(section_name, step_dict, type_map)
                print(f"    {i}. HF  {hf_summary}")
                for j, line in enumerate(ov_repr):
                    prefix = "       OV  " if j == 0 else "           "
                    print(f"{prefix}{line}")

        section_info[section_name] = {
            "hf_steps": hf_steps,
            "supported": len(unsupported) == 0,
            "unsupported_types": unsupported,
        }

    return section_info


def _get_type_map_for_section(section_name: str, parser_class) -> dict | None:
    """Return the type→handler map for a given section, or None if not map-based."""
    if section_name == "normalizer":
        return parser_class.normalizers_map
    elif section_name == "pre_tokenizer":
        return parser_class.pre_tokenization_map
    elif section_name == "post_processor":
        return parser_class.post_tokenization_map
    elif section_name == "decoder":
        return parser_class.decoding_map
    elif section_name == "model":
        return {"WordPiece": True, "BPE": True, "Unigram": True, "WordLevel": True}
    return None


def _try_map_step(section_name: str, step_dict: dict, type_map: dict | None) -> list[str]:
    """Try to map a single HF step dict to OV step(s). Returns list of formatted strings."""
    if type_map is None:
        return ["(no mapping available)"]

    step_type = step_dict.get("type", "<unknown>")

    if section_name == "model":
        model_step_names = {
            "WordPiece": "WordPieceTokenizationStep",
            "BPE": "BPETokenizationStep",
            "Unigram": "UnigramModelStep",
            "WordLevel": "VocabEncoderStep",
        }
        return [f"→ {model_step_names.get(step_type, '?')}(...)"]

    handler = type_map.get(step_type)
    if handler is None:
        return [f"{RED}⚠ UNSUPPORTED{RESET}"]

    try:
        result = handler(step_dict)
        if not isinstance(result, list):
            result = [result]
        return [f"→ {_format_ov_step(s)}" for s in result]
    except Exception as exc:
        return [f"→ {RED}<error: {exc}>{RESET}"]


# ── finalized pipeline display ────────────────────────────────────────────────


def step_show_finalized_pipeline(hf_tokenizer) -> dict:
    """
    Step 2b — run the actual parser + finalize() pipeline and show the resulting
    OV steps per stage. This reveals step merges (e.g. merge_regex_split_steps)
    that change the pipeline structure after the raw HF→OV mapping.

    Returns a dict with:
      {
        "pre_tokenization_before": [list of step reprs before finalize],
        "pre_tokenization_after":  [list of step reprs after finalize],
        "merge_occurred": bool,
      }
    """
    from copy import deepcopy

    from openvino_tokenizers.hf_parser import TransformersTokenizerPipelineParser
    from openvino_tokenizers.utils import TokenzierConversionParams

    params = TokenzierConversionParams()
    parser = TransformersTokenizerPipelineParser(hf_tokenizer, params)
    pipeline = parser.parse()

    from openvino_tokenizers.tokenizer_pipeline import RegexSplitStep as _RegexSplitStep

    # Snapshot pre-tokenization steps before finalization
    before = [_format_ov_step(s) for s in pipeline.pre_tokenization_steps]
    before_regex_count = sum(1 for s in pipeline.pre_tokenization_steps if isinstance(s, _RegexSplitStep))
    pre_finalization_norm_steps = list(pipeline.normalization_steps)
    pre_finalization_pre_tok_steps = list(pipeline.pre_tokenization_steps)

    # Finalize (triggers merge_regex_split_steps among other things)
    pipeline.finalize()

    after = [_format_ov_step(s) for s in pipeline.pre_tokenization_steps]
    after_regex_count = sum(1 for s in pipeline.pre_tokenization_steps if isinstance(s, _RegexSplitStep))

    # A merge occurred only when RegexSplitSteps were combined (not when
    # BytesToCharsStep was removed — that is expected during BPE finalization)
    merge_occurred = before_regex_count > after_regex_count

    print(f"\n  {BOLD}Finalized OV pre-tokenization steps:{RESET}")
    if not after:
        print(f"    (none)")
    else:
        for i, s in enumerate(after, 1):
            print(f"    {i}. {s}")

    if merge_occurred:
        print(
            f"\n  {YELLOW}⚠ Pre-tokenization merge: "
            f"{len(before)} steps → {len(after)} steps after finalization{RESET}"
        )
        print(f"    Before: {before}")
        print(f"    After:  {after}")
    elif before:
        print(f"  {GREEN}No step merges occurred{RESET}")

    return {
        "pre_tokenization_before": before,
        "pre_tokenization_after": after,
        "merge_occurred": merge_occurred,
        "finalized_pipeline": pipeline,
        "pre_finalization_norm_steps": pre_finalization_norm_steps,
        "pre_finalization_pre_tok_steps": pre_finalization_pre_tok_steps,
    }


# ── normalization diagnostics ─────────────────────────────────────────────────


def step_test_normalization(hf_tokenizer, tokenizer_json: dict, test_strings: list[str]) -> tuple[int, list[str]]:
    """
    Step 3 — test normalization steps individually and combined.
    Returns (number_of_failures, list_of_failing_step_types).
    """
    from .check_normalization import (
        _build_hf_normalizer,
        _build_ov_normalizer,
        _run_step_test,
    )
    from openvino_tokenizers.hf_parser import TransformersTokenizerPipelineParser

    normalizer = tokenizer_json.get("normalizer")
    if normalizer is None:
        _ok("No normalizer — skipping")
        return 0, []

    step_dicts = _get_section_steps(normalizer)
    if not step_dicts:
        _ok("No normalizer steps — skipping")
        return 0, []

    normalizers_map = TransformersTokenizerPipelineParser.normalizers_map
    total_failures = 0
    failing_types = []

    for step_dict in step_dicts:
        step_type = step_dict.get("type", "<unknown>")

        if step_type not in normalizers_map:
            _warn(f"  {BOLD}{step_type}{RESET}: unsupported — skipping")
            failing_types.append(step_type)
            continue

        try:
            ov_steps = normalizers_map[step_type](step_dict)
            if not isinstance(ov_steps, list):
                ov_steps = [ov_steps]
            ov_compiled = _build_ov_normalizer(ov_steps)
        except Exception as exc:
            _fail(f"  {BOLD}{step_type}{RESET}: failed to build OV model: {exc}")
            total_failures += 1
            failing_types.append(step_type)
            continue

        try:
            hf_norm = _build_hf_normalizer(step_dict)
        except Exception as exc:
            _fail(f"  {BOLD}{step_type}{RESET}: failed to build HF normalizer: {exc}")
            total_failures += 1
            failing_types.append(step_type)
            continue

        failures = _run_step_test(step_type, ov_compiled, hf_norm, test_strings)
        if failures:
            total_failures += failures
            failing_types.append(step_type)

    # Combined pipeline test
    if len(step_dicts) > 1 and hasattr(hf_tokenizer, "backend_tokenizer"):
        hf_combined = hf_tokenizer.backend_tokenizer.normalizer
        if hf_combined is not None:
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
                    if failures:
                        total_failures += failures
                        if "combined" not in failing_types:
                            failing_types.append("combined")
                except Exception as exc:
                    _warn(f"  Combined pipeline: could not run — {exc}")

    return total_failures, failing_types


# ── pre-tokenization diagnostics ─────────────────────────────────────────────


def _build_ov_pre_tokenizer(norm_steps, pre_tok_steps) -> object:
    """
    Build an isolated OV model that runs normalization + pre-tokenization steps.
    Uses *pre-finalization* steps so that transformations like BytesToCharsStep
    are still present (finalize() removes them and absorbs them into the
    tokenization model).

    Accepts a single string and returns a flat string tensor of split tokens.
    """
    from openvino import Core, Model, PartialShape, Type, op

    from openvino_tokenizers import _get_opset_factory
    from openvino_tokenizers.tokenizer_pipeline import TokenizerPipeline

    input_node = op.Parameter(Type.string, PartialShape(["?"]))
    input_node.set_friendly_name("string_input")
    output = _get_opset_factory("opset15").create(
        "StringTensorUnpack", input_node.outputs()
    ).outputs()

    # Normalization
    for step in norm_steps:
        output = step.get_ov_subgraph(output)

    # Add ragged dimension (same as TokenizerPipeline.add_ragged_dimension)
    output = TokenizerPipeline.add_ragged_dimension(output)

    # Pre-tokenization
    for step in pre_tok_steps:
        output = step.get_ov_subgraph(output)

    # Drop ragged dimension, keep only [begins, ends, chars] and pack to strings
    string_outputs = output[2:]  # skip ragged_begins, ragged_ends
    packed = _get_opset_factory("opset15").create(
        "StringTensorPack", string_outputs
    ).outputs()

    model = Model(packed, [input_node], "pre_tokenizer")
    return Core().compile_model(model)


def step_test_pre_tokenization(
    hf_tokenizer,
    finalized_pipeline,
    test_strings: list[str],
) -> tuple[int, list[str]]:
    """
    Step 4 — build an isolated OV pre-tokenizer model from the finalized
    pipeline, run it on each test string, and compare the split tokens with
    HF's ``backend_tokenizer.pre_tokenizer.pre_tokenize_str()``.

    Returns (number_of_failures, list_of_detail_strings).
    """
    if (
        not hasattr(hf_tokenizer, "backend_tokenizer")
        or hf_tokenizer.backend_tokenizer.pre_tokenizer is None
    ):
        _ok("No pre-tokenizer — skipping")
        return 0, []

    hf_pre_tokenizer = hf_tokenizer.backend_tokenizer.pre_tokenizer
    hf_normalizer = getattr(hf_tokenizer.backend_tokenizer, "normalizer", None)

    # Build finalized pipeline if not provided
    if finalized_pipeline is None:
        from openvino_tokenizers.hf_parser import TransformersTokenizerPipelineParser
        from openvino_tokenizers.utils import TokenzierConversionParams

        try:
            params = TokenzierConversionParams()
            parser = TransformersTokenizerPipelineParser(hf_tokenizer, params)
            parsed_pipeline = parser.parse()
            norm_steps = list(parsed_pipeline.normalization_steps)
            pre_tok_steps = list(parsed_pipeline.pre_tokenization_steps)
        except Exception as exc:
            _fail(f"Could not build pipeline for pre-tokenizer comparison: {exc}")
            return 1, [str(exc)]
    else:
        norm_steps = finalized_pipeline.get("pre_finalization_norm_steps", [])
        pre_tok_steps = finalized_pipeline.get("pre_finalization_pre_tok_steps", [])

    if not pre_tok_steps:
        _ok("No pre-tokenization steps in OV pipeline — skipping")
        return 0, []

    # Build isolated OV pre-tokenizer model
    try:
        ov_pre_tok = _build_ov_pre_tokenizer(norm_steps, pre_tok_steps)
    except Exception as exc:
        _fail(f"Could not build OV pre-tokenizer model: {exc}")
        return 1, [str(exc)]

    mismatches: list[tuple[str, str, str]] = []

    for s in test_strings:
        # Skip empty strings — the isolated OV model can't pack empty ragged output
        if not s:
            continue

        # HF: normalize then pre-tokenize (mirrors the pipeline order)
        try:
            normalized = hf_normalizer.normalize_str(s) if hf_normalizer else s
            hf_result = hf_pre_tokenizer.pre_tokenize_str(normalized)
            hf_tokens = [tok for tok, _span in hf_result]
        except Exception:
            continue

        # OV: run the compiled pre-tokenizer model
        try:
            ov_output = ov_pre_tok([s])
            ov_tokens = [str(t) for t in ov_output[0]]
        except Exception as exc:
            mismatches.append((s, str(hf_tokens), f"<OV raised {type(exc).__name__}: {exc}>"))
            continue

        if hf_tokens != ov_tokens:
            mismatches.append((s, str(hf_tokens), str(ov_tokens)))

    total = sum(1 for s in test_strings if s)
    if mismatches:
        passed = total - len(mismatches)
        _fail(f"Pre-tokenization: {passed}/{total} matched — {len(mismatches)} failed")
        for s, expected, actual in mismatches[:5]:
            print(f"\n    {YELLOW}Input:{RESET}    {_truncate(s)}", file=sys.stderr)
            print(f"    Expected: {expected}", file=sys.stderr)
            print(f"    Actual:   {actual}", file=sys.stderr)
    else:
        _ok(f"Pre-tokenization: all {total} strings matched")

    return len(mismatches), [f"{s}: expected {e}, got {a}" for s, e, a in mismatches]


# ── full pipeline comparison ──────────────────────────────────────────────────


def step_test_full_pipeline(hf_tokenizer, test_strings: list[str]) -> tuple[int, str]:
    """
    Step 5 — run the full pipeline (convert + compare) to find the first
    point of divergence.

    Returns (number_of_failures, failure_stage).
    """
    from openvino import Core
    from openvino_tokenizers import convert_tokenizer

    try:
        ov_tok_model, ov_detok_model = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    except Exception as exc:
        _fail(f"Conversion failed: {exc}")
        return 1, "conversion"

    core = Core()
    ov_tok = core.compile_model(ov_tok_model)
    ov_detok = core.compile_model(ov_detok_model)

    encode_failures = 0
    decode_failures = 0
    encode_mismatches = []
    decode_mismatches = []

    for s in test_strings:
        # Encode comparison
        hf_out = hf_tokenizer([s], return_tensors="np", truncation=True)
        ov_out = ov_tok([s])

        hf_ids = hf_out["input_ids"]
        try:
            ov_ids = ov_out["input_ids"]
        except (KeyError, RuntimeError):
            encode_failures += 1
            encode_mismatches.append((s, "missing input_ids output"))
            continue

        if not np.array_equal(hf_ids, ov_ids):
            encode_failures += 1
            encode_mismatches.append((
                s,
                f"HF: {hf_ids.reshape(-1).tolist()[:10]} vs OV: {ov_ids.reshape(-1).tolist()[:10]}",
            ))

        # Decode comparison
        hf_decoded = hf_tokenizer.batch_decode(hf_ids, skip_special_tokens=True)
        ov_decoded = ov_detok(hf_ids.astype("int32"))["string_output"].tolist()
        if hf_decoded != ov_decoded:
            decode_failures += 1
            decode_mismatches.append((s, f"HF: {repr(hf_decoded[0])} vs OV: {repr(ov_decoded[0])}"))

    total = len(test_strings)

    if encode_failures:
        _fail(f"Encode: {total - encode_failures}/{total} matched — {encode_failures} failed")
        for s, detail in encode_mismatches[:5]:  # limit output
            print(f"    {YELLOW}Input:{RESET} {_truncate(s)}", file=sys.stderr)
            print(f"    {detail}", file=sys.stderr)
    else:
        _ok(f"Encode: all {total} strings matched")

    if decode_failures:
        _fail(f"Decode: {total - decode_failures}/{total} matched — {decode_failures} failed")
        for s, detail in decode_mismatches[:5]:
            print(f"    {YELLOW}Input:{RESET} {_truncate(s)}", file=sys.stderr)
            print(f"    {detail}", file=sys.stderr)
    else:
        _ok(f"Decode: all {total} strings matched")

    if encode_failures and decode_failures:
        return encode_failures + decode_failures, "encode+decode"
    elif encode_failures:
        return encode_failures, "encode"
    elif decode_failures:
        return decode_failures, "decode"
    return 0, "none"


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
        prog="openvino_tokenizers diagnose",
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
    _configure_parser(parser)
    return parser


def run(args) -> None:
    total_steps = 5
    exit_code = 0

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
    _step(2, total_steps, "Mapping tokenizer.json pipeline → OV steps")
    tokenizer_json = _load_tokenizer_json(hf_tokenizer)
    if tokenizer_json is None:
        _fail("Could not extract tokenizer.json (not a fast tokenizer?)")
        sys.exit(1)

    from openvino_tokenizers.hf_parser import TransformersTokenizerPipelineParser

    try:
        section_info = step_print_pipeline_map(tokenizer_json, TransformersTokenizerPipelineParser)
    except Exception:
        _fail("Failed to map pipeline:")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    all_unsupported = []
    for sec_name, info in section_info.items():
        all_unsupported.extend(info["unsupported_types"])

    if all_unsupported:
        print(f"\n  {RED}Unsupported types:{RESET} {', '.join(all_unsupported)}")
        exit_code = 1
    else:
        print(f"\n  {GREEN}All pipeline types are supported{RESET}")
    # Show the finalized pipeline (reveals step merges)
    finalized_info = {"merge_occurred": False, "finalized_pipeline": None}
    try:
        finalized_info = step_show_finalized_pipeline(hf_tokenizer)
        if finalized_info["merge_occurred"]:
            exit_code = 1
    except Exception:
        _warn("Could not show finalized pipeline:")
        traceback.print_exc(file=sys.stderr)
    # ── Step 3 ────────────────────────────────────────────────────────────────
    _step(3, total_steps, "Testing normalization steps")
    try:
        norm_failures, norm_failing_types = step_test_normalization(
            hf_tokenizer, tokenizer_json, ALL_TEST_STRINGS
        )
        if norm_failures:
            exit_code = 1
    except Exception:
        _fail("Normalization testing raised an unexpected exception:")
        traceback.print_exc(file=sys.stderr)
        norm_failures = 1
        norm_failing_types = ["<exception>"]
        exit_code = 1

    # ── Step 4 ────────────────────────────────────────────────────────────────
    _step(4, total_steps, "Testing pre-tokenization")
    try:
        pre_tok_failures, pre_tok_details = step_test_pre_tokenization(
            hf_tokenizer, finalized_info if finalized_info.get("finalized_pipeline") else None, ALL_TEST_STRINGS
        )
        if pre_tok_failures:
            exit_code = 1
    except Exception:
        _fail("Pre-tokenization testing raised an unexpected exception:")
        traceback.print_exc(file=sys.stderr)
        pre_tok_failures = 1
        pre_tok_details = ["<exception>"]
        exit_code = 1

    # ── Step 5 ────────────────────────────────────────────────────────────────
    _step(5, total_steps, "Full pipeline comparison")
    try:
        full_failures, failure_stage = step_test_full_pipeline(hf_tokenizer, ALL_TEST_STRINGS)
        if full_failures:
            exit_code = 1
    except Exception:
        _fail("Full pipeline comparison raised an unexpected exception:")
        traceback.print_exc(file=sys.stderr)
        full_failures = 1
        failure_stage = "<exception>"
        exit_code = 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"{BOLD}Diagnosis Summary{RESET}")
    print(f"{'─' * 60}")

    merge_occurred = finalized_info.get("merge_occurred", False)

    # Determine root cause location
    if all_unsupported:
        root_cause = "python"
        suggestion = "tokenizer-fix-python"
        description = f"Unsupported types need handlers in hf_parser.py: {', '.join(all_unsupported)}"
    elif pre_tok_failures and not norm_failures:
        root_cause = "python"
        suggestion = "tokenizer-fix-python"
        if merge_occurred:
            description = (
                "Pre-tokenization mismatch after RegexSplitStep merge in finalize(); "
                "merge_regex_split_steps likely combines incompatible patterns"
            )
        else:
            description = "Pre-tokenization mismatch; check pre-tokenizer step mapping in hf_parser.py"
    elif norm_failures and full_failures:
        root_cause = "cpp"
        suggestion = "tokenizer-fix-cpp"
        description = f"Normalization mismatches in: {', '.join(norm_failing_types)}"
    elif full_failures and not norm_failures:
        root_cause = "python"
        suggestion = "tokenizer-fix-python"
        description = f"Full pipeline diverges at {failure_stage} stage; normalization is OK"
    elif norm_failures:
        root_cause = "cpp"
        suggestion = "tokenizer-fix-cpp"
        description = f"Normalization mismatches in: {', '.join(norm_failing_types)}"
    else:
        root_cause = "none"
        suggestion = "none"
        description = "All stages matched"

    affected_stages = []
    if all_unsupported:
        for sec_name, info in section_info.items():
            if info["unsupported_types"]:
                affected_stages.append(sec_name)
    if norm_failures:
        affected_stages.append("normalization")
    if pre_tok_failures:
        affected_stages.append("pre_tokenization")
    if full_failures and failure_stage != "none":
        if "encode" in failure_stage:
            affected_stages.append("encode")
        if "decode" in failure_stage:
            affected_stages.append("decode")

    # Deduplicate while preserving order
    seen = set()
    affected_stages = [s for s in affected_stages if not (s in seen or seen.add(s))]

    print(f"  root_cause_location: {root_cause}")
    print(f"  affected_stages: {affected_stages}")
    print(f"  unsupported_types: {all_unsupported}")
    print(f"  normalization_failures: {norm_failures}")
    print(f"  pre_tokenization_failures: {pre_tok_failures}")
    print(f"  full_pipeline_failures: {full_failures}")
    print(f"  description: {description}")
    print(f"  suggested_fix_skill: {suggestion}")

    sys.exit(exit_code)
