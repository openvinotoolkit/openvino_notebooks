# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: MIT

"""Extract speaker embeddings with the OpenVINO IR of pyannote/embedding.

Convert the model first:
    python convert_to_openvino.py

Then run:
    # Single file
    python infer_openvino.py audio.wav

    # Compare two speakers (cosine distance; smaller = same speaker)
    python infer_openvino.py speaker1.wav speaker2.wav --device CPU
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations

import numpy as np
import openvino as ov

from common import cosine_distance, load_audio, to_model_input


def build_compiled_model(
    model_path: str,
    device: str,
    precision: str | None = None,
    num_samples: int | None = None,
):
    core = ov.Core()
    model = core.read_model(model_path)
    if num_samples is not None:
        # A static input shape is required for a forced f32 precision on GPU:
        # the dynamic time axis makes the GPU plugin fail to build the stats-
        # pooling (Power) kernel. CPU works either way.
        model.reshape({0: ov.PartialShape([1, 1, num_samples])})
    config = {}
    if precision:
        # Pin the compute precision (e.g. "f32") so GPU results match CPU/PyTorch.
        # Without this, the GPU plugin may default to f16 for speed.
        config["INFERENCE_PRECISION_HINT"] = precision
    return core.compile_model(model, device, config)


def embed(compiled, path: str) -> np.ndarray:
    waveform = load_audio(path)
    model_input = to_model_input(waveform)
    result = compiled(model_input)
    embedding = result[compiled.output(0)][0]
    return embedding


def benchmark(compiled, path: str, device: str, runs: int) -> None:
    """Print mean/min inference latency (ms) over ``runs`` timed calls."""
    model_input = to_model_input(load_audio(path))
    compiled(model_input)  # warmup
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        compiled(model_input)
        times.append((time.perf_counter() - start) * 1000.0)
    times_arr = np.asarray(times)
    print(
        f"{path}: inference {times_arr.mean():.2f} ms "
        f"(min {times_arr.min():.2f}, runs={runs}, device={device})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", help="one or two audio files")
    parser.add_argument(
        "--model", default="models/pyannote_embedding.xml", help="path to the IR .xml"
    )
    parser.add_argument(
        "--device", default="CPU", help="OpenVINO device: CPU, GPU, NPU, AUTO ..."
    )
    parser.add_argument(
        "--precision",
        default=None,
        choices=["f32", "f16"],
        help="inference precision hint (e.g. f32 for closer GPU/CPU parity); "
        "default lets the plugin choose (GPU often picks f16)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=0,
        help="if >0, benchmark inference latency in ms over this many runs "
        "(after a warmup) instead of just printing embeddings",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="when 3+ files are provided, also print cosine distance for all pairs",
    )
    args = parser.parse_args()

    print(f"Available devices: {ov.Core().available_devices}")
    precision_note = f" (precision={args.precision})" if args.precision else ""
    print(f"Loading IR {args.model} on {args.device}{precision_note} ...")

    if args.runs:
        if args.precision:
            compiled_by_len: dict[int, object] = {}
            for path in args.audio:
                num_samples = to_model_input(load_audio(path)).shape[-1]
                if num_samples not in compiled_by_len:
                    compiled_by_len[num_samples] = build_compiled_model(
                        args.model, args.device, args.precision, num_samples
                    )
                benchmark(compiled_by_len[num_samples], path, args.device, args.runs)
        else:
            compiled = build_compiled_model(args.model, args.device)
            for path in args.audio:
                benchmark(compiled, path, args.device, args.runs)
        return

    if args.precision:
        # Forced precision needs a static shape; compile once per input length.
        compiled_by_len: dict[int, object] = {}
        embeddings = {}
        for path in args.audio:
            model_input = to_model_input(load_audio(path))
            num_samples = model_input.shape[-1]
            if num_samples not in compiled_by_len:
                compiled_by_len[num_samples] = build_compiled_model(
                    args.model, args.device, args.precision, num_samples
                )
            compiled = compiled_by_len[num_samples]
            embeddings[path] = compiled(model_input)[compiled.output(0)][0]
    else:
        compiled = build_compiled_model(args.model, args.device)
        embeddings = {path: embed(compiled, path) for path in args.audio}

    for path, emb in embeddings.items():
        print(f"{path}: shape={emb.shape} norm={np.linalg.norm(emb):.4f}")

    if len(args.audio) == 2:
        a, b = args.audio
        dist = cosine_distance(embeddings[a], embeddings[b])
        print(f"\nCosine distance({a}, {b}) = {dist:.4f}")
        print("(smaller distance => more likely the same speaker)")
    elif args.all_pairs and len(args.audio) >= 2:
        print("\nPairwise cosine distances:")
        for a, b in combinations(args.audio, 2):
            dist = cosine_distance(embeddings[a], embeddings[b])
            print(f"Cosine distance({a}, {b}) = {dist:.4f}")
        print("(smaller distance => more likely the same speaker)")


if __name__ == "__main__":
    main()
