# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: MIT

"""Run the pyannote speaker-diarization pipeline with OpenVINO acceleration.

Same pipeline as ``run_diarization.py`` (community-1: segmentation + embedding +
clustering), but the two heavy neural blocks run as **OpenVINO IR** while
pyannote keeps orchestrating the pipeline (windowing, clustering, PLDA) in
Python.

What is accelerated with OpenVINO:
    * segmentation model  (PyanNet)            -- whole model -> OV IR
    * embedding model     (WeSpeakerResNet34)  -- ResNet part -> OV IR
                                                  (fbank stays in PyTorch:
                                                   it uses vmap+Kaldi which
                                                   TorchScript cannot trace)
Clustering / PLDA stay on CPU/numpy (they are not neural nets).

The IR is produced by ``export_pyann.py`` (run it once). This script only loads
the cached IR from ``ov_models/`` -- it does not convert anything.

The pipeline is gated -- accept the conditions once on Hugging Face and log in
with ``huggingface-cli login`` (or pass ``--token``).

Examples:
    python export_pyann.py                 # one-time: create the IR
    python run_diarization_ov.py audio.wav
    python run_diarization_ov.py audio.wav --num-speakers 2 --device CPU
    python run_diarization_ov.py audio.wav --rttm out.rttm
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import openvino as ov
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from export_pyann import (
    DEFAULT_OUTPUT_DIR,
    EMBEDDING_XML,
    PIPELINE_ID,
    SEGMENTATION_XML,
)


def _bucket_size(b: int, max_batch: int) -> int:
    """Smallest power-of-two >= b, capped at max_batch (batch bucketing).

    Padding every call up to the max batch wastes a lot of compute on the iGPU
    (cost scales with batch). Bucketing pads only to the next power of two, so a
    batch of 2 runs as 2 (not 32).
    """
    size = 1
    while size < b:
        size *= 2
    return min(size, max_batch)


def accelerate_segmentation(
    pipeline: Pipeline, core: ov.Core, ir_dir: Path, device: str, static: bool
) -> None:
    """Route the segmentation model's forward through the pre-exported OV IR.

    On GPU, dynamic input shapes are extremely slow, so ``static=True`` reshapes
    the IR to fixed batch buckets (padding each call up to the next power-of-two
    bucket) -- this is ~13x faster on the Arc dGPU and avoids over-padding on the
    iGPU.
    """
    model = pipeline._segmentation.model
    xml_path = ir_dir / SEGMENTATION_XML
    if not xml_path.exists():
        raise SystemExit(
            f"Missing IR '{xml_path}'. Export it first:\n"
            f"    python export_pyann.py --output-dir {xml_path.parent}"
        )

    if not static:
        compiled = core.compile_model(core.read_model(xml_path), device)
        out_port = compiled.output(0)

        def forward(waveforms, *args, **kwargs):
            data = waveforms.detach().cpu().numpy().astype(np.float32)
            return torch.from_numpy(compiled(data)[out_port])

        model.forward = forward
        return

    max_batch = pipeline._segmentation.batch_size
    buckets: dict = {}  # bucket_size -> (compiled_model, out_port)

    def forward(waveforms, *args, **kwargs):
        data = waveforms.detach().cpu().numpy().astype(np.float32)
        b = data.shape[0]
        size = _bucket_size(b, max_batch)
        if size not in buckets:
            ov_model = core.read_model(xml_path)
            ov_model.reshape([size, data.shape[1], data.shape[2]])
            compiled = core.compile_model(ov_model, device)
            buckets[size] = (compiled, compiled.output(0))
        compiled, out_port = buckets[size]
        if b < size:
            pad = np.zeros((size - b, *data.shape[1:]), np.float32)
            data = np.concatenate([data, pad], axis=0)
        result = compiled(data)[out_port][:b]
        return torch.from_numpy(result.copy())

    model.forward = forward


def accelerate_embedding(
    pipeline: Pipeline, core: ov.Core, ir_dir: Path, device: str, static: bool
) -> None:
    """Route the embedding ResNet through OV IR; keep fbank in PyTorch.

    On GPU, ``static=True`` reshapes the ResNet IR to fixed batch buckets + frame
    count (padding each call up to the next power-of-two bucket) -- this is ~150x
    faster on the Arc dGPU and avoids over-padding on the iGPU. Calls without 2-D
    weights (e.g. calibration) always fall back to PyTorch.
    """
    model = pipeline._embedding.model_
    resnet = model.resnet  # unchanged torch module (fallback path)
    xml_path = ir_dir / EMBEDDING_XML
    if not xml_path.exists():
        raise SystemExit(
            f"Missing IR '{xml_path}'. Export it first:\n"
            f"    python export_pyann.py --output-dir {xml_path.parent}"
        )

    if not static:
        compiled = core.compile_model(core.read_model(xml_path), device)
        out_port = compiled.output(0)

        def forward(waveforms, weights=None, *args, **kwargs):
            fbank = model.compute_fbank(waveforms.detach().cpu())
            if weights is None or weights.ndim != 2:
                return resnet(fbank, weights=weights)[1]
            result = compiled(
                (
                    fbank.numpy().astype(np.float32),
                    weights.detach().cpu().numpy().astype(np.float32),
                )
            )[out_port]
            return torch.from_numpy(result)

        model.forward = forward
        return

    max_batch = pipeline.embedding_batch_size
    buckets: dict = {}  # (bucket_size, frames, wframes) -> (compiled, out_port)

    def forward(waveforms, weights=None, *args, **kwargs):
        fbank = model.compute_fbank(waveforms.detach().cpu())
        if weights is None or weights.ndim != 2:
            return resnet(fbank, weights=weights)[1]

        fb = fbank.numpy().astype(np.float32)
        wt = weights.detach().cpu().numpy().astype(np.float32)
        b, frames, mels = fb.shape
        wframes = wt.shape[1]
        size = _bucket_size(b, max_batch)

        key = (size, frames, wframes)
        if key not in buckets:
            ov_model = core.read_model(xml_path)
            ov_model.reshape({0: [size, frames, mels], 1: [size, wframes]})
            compiled = core.compile_model(ov_model, device)
            buckets[key] = (compiled, compiled.output(0))
        compiled, out_port = buckets[key]

        if b < size:
            fb = np.concatenate(
                [fb, np.zeros((size - b, frames, mels), np.float32)], axis=0
            )
            # pad weights with ones (not zeros) so stats-pooling never divides by 0
            wt = np.concatenate(
                [wt, np.ones((size - b, wframes), np.float32)], axis=0
            )
        result = compiled((fb, wt))[out_port][:b]
        return torch.from_numpy(result.copy())

    model.forward = forward


def build_pipeline(
    token: str | None, device: str, ir_dir: Path, static: bool
) -> Pipeline:
    pipeline = Pipeline.from_pretrained(PIPELINE_ID, token=token)
    if pipeline is None:
        raise SystemExit(
            f"Failed to load '{PIPELINE_ID}'. Accept the user conditions on "
            "Hugging Face and run `huggingface-cli login` (or pass --token)."
        )
    pipeline.to(torch.device("cpu"))  # pyannote orchestration stays on CPU torch

    core = ov.Core()
    accelerate_segmentation(pipeline, core, ir_dir, device, static)
    accelerate_embedding(pipeline, core, ir_dir, device, static)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="path to an audio file (wav, flac, ...)")
    parser.add_argument("--token", default=None, help="HF access token override")
    parser.add_argument(
        "--device", default="CPU", help="OpenVINO device: CPU, GPU.0, GPU.1"
    )
    parser.add_argument(
        "--ir-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="folder with the exported IR (default: ./ov_models)",
    )
    parser.add_argument(
        "--static",
        choices=["auto", "on", "off"],
        default="auto",
        help="static-shape + batch padding (much faster on GPU). "
        "'auto' = on for GPU devices, off for CPU.",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None, help="exact number of speakers"
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None, help="minimum number of speakers"
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None, help="maximum number of speakers"
    )
    parser.add_argument(
        "--rttm", default=None, help="optional path to write the result as RTTM"
    )
    args = parser.parse_args()

    if args.static == "auto":
        static = args.device.upper().startswith("GPU")
    else:
        static = args.static == "on"

    pipeline = build_pipeline(args.token, args.device, Path(args.ir_dir), static)

    constraints: dict[str, int] = {}
    if args.num_speakers is not None:
        constraints["num_speakers"] = args.num_speakers
    if args.min_speakers is not None:
        constraints["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        constraints["max_speakers"] = args.max_speakers

    import soundfile as sf

    duration = sf.info(args.audio).duration
    print(f"# audio: {args.audio}  duration={duration:.1f}s")

    start = time.perf_counter()
    with ProgressHook() as hook:
        output = pipeline(args.audio, hook=hook, **constraints)
    elapsed = time.perf_counter() - start

    diarization = output.speaker_diarization

    print(f"\n# diarization for {args.audio} (OpenVINO {args.device}, {elapsed:.1f}s)")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:6.1f}s stop={turn.end:6.1f}s {speaker}")

    speakers = diarization.labels()
    print(f"\n# {len(speakers)} speaker(s): {', '.join(speakers)}")

    if args.rttm:
        with open(args.rttm, "w") as f:
            diarization.write_rttm(f)
        print(f"# wrote RTTM -> {args.rttm}")


if __name__ == "__main__":
    main()
