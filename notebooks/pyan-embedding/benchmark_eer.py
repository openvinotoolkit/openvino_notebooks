# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: MIT

"""Reproduce the VoxCeleb1 speaker-verification EER for pyannote/embedding.

The model card claims 2.8% EER on the VoxCeleb1 test set using cosine distance
directly (no VAD, no PLDA). This script embeds every unique clip referenced by
the trial list once, scores each trial pair by cosine similarity, and computes
the Equal Error Rate.

Backends:
    # PyTorch (reference)
    python benchmark_eer.py --backend pytorch --device cpu
    python benchmark_eer.py --backend pytorch --device xpu

    # OpenVINO
    python benchmark_eer.py --backend openvino --device CPU
    python benchmark_eer.py --backend openvino --device GPU.0
    python benchmark_eer.py --backend openvino --device GPU.0 --precision f32

Data layout (defaults, relative to the current working directory):
    ./test_audio/veri_test.txt
    ./test_audio/vox1/wav/<id>/<clip>/<utt>.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import MODEL_ID, load_audio, to_model_input


def load_trials(path: Path, limit: int | None = None):
    """Return a list of (label, path_a, path_b) from the trial file."""
    pairs = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 3:
                continue
            label, a, b = parts
            pairs.append((int(label), a, b))
            if limit is not None and len(pairs) >= limit:
                break
    return pairs


def compute_eer(scores: np.ndarray, labels: np.ndarray):
    """Equal Error Rate (%) and its threshold. Higher score = same speaker."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer * 100.0, float(thresholds[idx])


def embed_pytorch(files, wav_root: Path, device: str):
    import torch
    from pyannote.audio import Model

    model = Model.from_pretrained(MODEL_ID).eval().to(torch.device(device))
    embeddings = {}
    with torch.no_grad():
        for i, rel in enumerate(files, 1):
            waveform = load_audio(str(wav_root / rel))
            tensor = torch.from_numpy(to_model_input(waveform)).to(torch.device(device))
            embeddings[rel] = model(tensor).cpu().numpy()[0]
            if i % 200 == 0:
                print(f"  embedded {i}/{len(files)}", flush=True)
    return embeddings


def embed_openvino(
    files,
    wav_root: Path,
    device: str,
    precision: str | None,
    performance_hint: str | None,
    cache_dir: str | None,
):
    import openvino as ov

    core = ov.Core()
    if cache_dir:
        # Persist compiled kernels across runs to reduce startup/reshape overhead.
        core.set_property({"CACHE_DIR": cache_dir})
    config = {}
    if precision:
        config["INFERENCE_PRECISION_HINT"] = precision
    if performance_hint and performance_hint != "auto":
        config["PERFORMANCE_HINT"] = performance_hint.upper()

    def compile_dynamic():
        model = core.read_model("models/pyannote_embedding.xml")
        return core.compile_model(model, device, config)

    def compile_static(num_samples: int):
        # f32 on GPU needs a static shape (the dynamic time axis makes the GPU
        # plugin fail to build the stats-pooling Power kernel). Compile once per
        # unique clip length and cache.
        model = core.read_model("models/pyannote_embedding.xml")
        model.reshape({0: ov.PartialShape([1, 1, num_samples])})
        return core.compile_model(model, device, config)

    # CPU uses static shapes for reproducible results regardless of whether the
    # caller passes an explicit FP32 precision hint. Cache by length so clips
    # with identical sample counts reuse their compiled model.
    static = device.upper() == "CPU" or precision == "f32"
    compiled_dynamic = None if static else compile_dynamic()
    compiled_by_len: dict[int, object] = {}

    embeddings = {}
    for i, rel in enumerate(files, 1):
        waveform = load_audio(str(wav_root / rel))
        model_input = to_model_input(waveform)
        if static:
            num_samples = model_input.shape[-1]
            compiled = compiled_by_len.get(num_samples)
            if compiled is None:
                compiled = compile_static(num_samples)
                compiled_by_len[num_samples] = compiled
        else:
            compiled = compiled_dynamic
        embeddings[rel] = compiled(model_input)[compiled.output(0)][0]
        if i % 200 == 0:
            print(f"  embedded {i}/{len(files)}", flush=True)
    return embeddings


def cosine_similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) arrays."""
    a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["pytorch", "openvino"], default="openvino")
    parser.add_argument("--device", default="CPU", help="cpu/xpu or CPU/GPU.0/GPU.1")
    parser.add_argument("--precision", default=None, choices=["f32", "f16"])
    parser.add_argument(
        "--ov-performance",
        default="auto",
        choices=["auto", "latency", "throughput"],
        help="OpenVINO PERFORMANCE_HINT (auto picks latency for this per-clip benchmark)",
    )
    parser.add_argument(
        "--ov-cache-dir",
        default=".ov_cache",
        help="OpenVINO cache directory for compiled kernels (set empty to disable)",
    )
    parser.add_argument(
        "--trials", default="./test_audio/veri_test.txt", help="trial pairs file"
    )
    parser.add_argument(
        "--wav-root", default="./test_audio/vox1/wav", help="root of the wav files"
    )
    parser.add_argument("--limit", type=int, default=None, help="limit #pairs (debug)")
    args = parser.parse_args()

    trials = load_trials(Path(args.trials), args.limit)
    files = sorted({p for _, a, b in trials for p in (a, b)})
    print(f"{len(trials)} pairs, {len(files)} unique clips")
    print(f"backend={args.backend} device={args.device} precision={args.precision}")

    wav_root = Path(args.wav_root)
    t0 = time.time()
    if args.backend == "pytorch":
        embeddings = embed_pytorch(files, wav_root, args.device)
    else:
        perf = args.ov_performance
        if perf == "auto":
            # For this benchmark we run one clip at a time with variable lengths.
            # Throughput mode helps batched pipelines, but usually hurts this
            # serial per-clip workload (especially on GPU), so default to latency.
            perf = "latency"
        cache_dir = args.ov_cache_dir.strip() if args.ov_cache_dir else None
        cache_dir = cache_dir or None
        print(f"openvino_perf={perf} cache_dir={cache_dir}")
        embeddings = embed_openvino(
            files,
            wav_root,
            args.device,
            args.precision,
            perf,
            cache_dir,
        )
    embed_time = time.time() - t0
    print(f"embedded {len(files)} clips in {embed_time:.1f}s")

    emb_a = np.stack([embeddings[a] for _, a, _ in trials])
    emb_b = np.stack([embeddings[b] for _, _, b in trials])
    scores = cosine_similarity_matrix(emb_a, emb_b)
    labels = np.array([label for label, _, _ in trials])

    eer, threshold = compute_eer(scores, labels)
    print(f"\n=== EER = {eer:.2f}%  (threshold cos-sim = {threshold:.4f}) ===")
    print("(model card claims 2.8% on the full VoxCeleb1 test set)")


if __name__ == "__main__":
    main()
