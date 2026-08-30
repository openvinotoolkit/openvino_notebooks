# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: MIT

"""Score diarization DER on Debug or VoxConverse.

By default, this script evaluates on the bundled ``tests/data`` Debug protocol
shipped with pyannote-audio. It can also evaluate VoxConverse v0.3 once the
dataset is downloaded locally.

Examples:
    # PyTorch CPU on Debug test split
    python score_der.py --dataset debug --backend cpu --subset test

    # OpenVINO CPU on Debug development split, forcing 2 speakers
    python score_der.py --dataset debug --backend ov-cpu --subset development --num-speakers 2

    # PyTorch XPU (Arc dGPU) on VoxConverse test split
    python score_der.py --dataset voxconverse --backend xpu --subset test \
        --vox-root /path/to/voxconverse
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

from pyannote.database import FileFinder, registry
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from run_diarization import build_pipeline as build_torch_pipeline, sync_device

PROTOCOL = "Debug.SpeakerDiarization.Debug"


def _resolve_subset_name(dataset: str, subset: str) -> str:
    if dataset == "debug":
        return subset
    # VoxConverse naming: dev/test in official repo.
    if subset == "development":
        return "dev"
    if subset == "test":
        return "test"
    raise ValueError(
        "VoxConverse supports only development|test subsets "
        "(mapped to dev|test)."
    )


def iter_subset(protocol, subset: str):
    if subset == "development":
        yield from protocol.development()
    elif subset == "test":
        yield from protocol.test()
    elif subset == "train":
        yield from protocol.train()
    else:
        raise ValueError(f"Unknown subset: {subset}")


def _build_audio_index(search_roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for wav_path in root.rglob("*.wav"):
            wav_resolved = wav_path.resolve()
            if wav_resolved in seen:
                continue
            seen.add(wav_resolved)
            index.setdefault(wav_path.stem, []).append(wav_path)
    return index


def _normalize_search_roots(roots: list[Path]) -> list[Path]:
    """Remove duplicates and nested roots to avoid scanning same WAV twice."""
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)

    # Keep broader roots first; skip descendants of already kept roots.
    ordered = sorted(unique, key=lambda p: len(str(p)))
    kept: list[Path] = []
    for candidate in ordered:
        if any(parent == candidate or parent in candidate.parents for parent in kept):
            continue
        kept.append(candidate)
    return kept


def _candidate_audio_roots(vox_root: Path, vox_audio_root: Path | None) -> list[Path]:
    if vox_audio_root is not None:
        return _normalize_search_roots([vox_audio_root])

    # Common layouts across local/manual downloads.
    roots = [
        vox_root,
        vox_root / "audio",
        vox_root / "wav",
        vox_root / "dev_wav",
        vox_root / "test_wav",
        vox_root / "voxconverse_dev_wav",
        vox_root / "voxconverse_test_wav",
        vox_root / "voxconverse_dev_wav" / "dev",
        vox_root / "voxconverse_test_wav" / "test",
    ]
    return _normalize_search_roots(roots)


def load_voxconverse_subset(
    subset: str,
    vox_root: Path,
    vox_audio_root: Path | None,
) -> list[dict]:
    split = _resolve_subset_name("voxconverse", subset)
    rttm_dir = vox_root / split
    if not rttm_dir.exists():
        raise SystemExit(
            f"Missing VoxConverse RTTM folder: {rttm_dir}\n"
            "Expected a 'dev/' or 'test/' directory containing .rttm files."
        )

    annotations: dict = {}
    for rttm_path in sorted(rttm_dir.glob("*.rttm")):
        annotations.update(load_rttm(rttm_path))
    if not annotations:
        raise SystemExit(f"No RTTM files found in {rttm_dir}")

    search_roots = _candidate_audio_roots(vox_root, vox_audio_root)
    audio_index = _build_audio_index(search_roots)

    files: list[dict] = []
    missing_audio: list[str] = []
    duplicate_audio: list[str] = []
    for uri, annotation in sorted(annotations.items()):
        candidates = audio_index.get(uri, [])
        if not candidates:
            missing_audio.append(uri)
            continue
        if len(candidates) > 1:
            duplicate_audio.append(uri)
        chosen = candidates[0]
        files.append(
            {
                "uri": uri,
                "audio": str(chosen),
                "annotation": annotation,
                "annotated": annotation.get_timeline().support(),
            }
        )

    if missing_audio:
        preview = ", ".join(missing_audio[:8])
        more = "" if len(missing_audio) <= 8 else f" ... (+{len(missing_audio) - 8} more)"
        raise SystemExit(
            "Could not find WAV files for some RTTM uris.\n"
            f"Missing: {preview}{more}\n"
            "Provide --vox-audio-root if audio is outside --vox-root."
        )

    if duplicate_audio:
        print(
            "# warning: multiple WAV matches found for some uris; "
            "using first match for: "
            + ", ".join(duplicate_audio[:8])
            + (" ..." if len(duplicate_audio) > 8 else "")
        )

    return files


def build_backend(backend: str, token: str | None, ir_dir: Path):
    backend = backend.lower()
    if backend == "cpu":
        return build_torch_pipeline(token, "cpu")
    if backend == "xpu":
        return build_torch_pipeline(token, "xpu")
    if backend == "ov-cpu":
        from run_diarization_ov import build_pipeline as build_ov_pipeline

        return build_ov_pipeline(token, "CPU", ir_dir, static=False)
    if backend in {"ov-gpu0", "ov-gpu"}:
        from run_diarization_ov import build_pipeline as build_ov_pipeline

        return build_ov_pipeline(token, "GPU.0", ir_dir, static=True)
    if backend == "ov-gpu1":
        from run_diarization_ov import build_pipeline as build_ov_pipeline

        return build_ov_pipeline(token, "GPU.1", ir_dir, static=True)
    raise ValueError(
        f"Unknown backend '{backend}'. Use cpu, xpu, ov-cpu, ov-gpu (or ov-gpu0), or ov-gpu1."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="debug",
        choices=["debug", "voxconverse"],
        help="dataset to evaluate",
    )
    parser.add_argument(
        "--backend",
        default="cpu",
        choices=["cpu", "xpu", "ov-cpu", "ov-gpu", "ov-gpu0", "ov-gpu1"],
        help="backend to evaluate",
    )
    parser.add_argument(
        "--subset",
        default="test",
        choices=["train", "development", "test"],
        help="split to score (for VoxConverse use development|test)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="force an exact speaker count (else auto-count)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF access token override for gated models",
    )
    parser.add_argument(
        "--ir-dir",
        default=str(Path(__file__).resolve().parent / "ov_models"),
        help="folder containing exported OpenVINO IR",
    )
    parser.add_argument(
        "--vox-root",
        default=None,
        help="VoxConverse root folder containing dev/ and test/ RTTM dirs",
    )
    parser.add_argument(
        "--vox-audio-root",
        default=None,
        help="optional explicit root folder for VoxConverse WAV files",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only header + TOTAL line (skip per-file lines)",
    )
    args = parser.parse_args()

    if args.dataset == "voxconverse" and args.subset == "train":
        raise SystemExit("VoxConverse has no train split in this scorer.")

    files: list[dict]
    subset_name = _resolve_subset_name(args.dataset, args.subset)
    if args.dataset == "debug":
        registry.load_database(Path(__file__).resolve().parents[1] / "tests/data/database.yml")
        protocol = registry.get_protocol(
            PROTOCOL, preprocessors={"audio": FileFinder()}
        )
        files = list(iter_subset(protocol, args.subset))
    else:
        if args.vox_root is None:
            raise SystemExit(
                "--vox-root is required for --dataset voxconverse "
                "(path containing dev/ and test/ RTTM dirs)."
            )
        files = load_voxconverse_subset(
            subset=args.subset,
            vox_root=Path(args.vox_root),
            vox_audio_root=Path(args.vox_audio_root) if args.vox_audio_root else None,
        )

    pipeline = build_backend(args.backend, args.token, Path(args.ir_dir))
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

    total_elapsed = 0.0
    print(
        f"# dataset={args.dataset} backend={args.backend} "
        f"subset={subset_name} files={len(files)}"
    )

    for current_file in files:
        start = time.perf_counter()
        output = pipeline(current_file, num_speakers=args.num_speakers)
        if args.backend == "cpu":
            sync_device("cpu")
        elif args.backend == "xpu":
            sync_device("xpu")
        elapsed = time.perf_counter() - start
        total_elapsed += elapsed

        hypothesis = output.speaker_diarization
        reference = current_file["annotation"]
        uem = current_file["annotated"]
        der = metric(reference, hypothesis, uem=uem, uri=current_file["uri"])
        if not args.summary_only:
            print(
                f"# {current_file['uri']}: der={100.0 * der:.1f}% "
                f"time={elapsed:.1f}s speakers={len(hypothesis.labels())}"
            )

    report = metric.report(display=False)
    total = report.loc["TOTAL", ("diarization error rate", "%")]
    print(f"# TOTAL DER={float(total):.1f}%  total_time={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
