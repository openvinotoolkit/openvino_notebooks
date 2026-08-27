# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: MIT

"""Run the full pyannote speaker-diarization pipeline on CPU.

Uses the open-source ``pyannote/speaker-diarization-community-1`` pipeline
(segmentation + embedding + clustering) to answer "who spoke when".

The pipeline is gated -- accept the conditions once on Hugging Face and log in
with ``huggingface-cli login`` (already done on this machine). A token can also
be passed explicitly with ``--token``.

Examples:
    # Diarize a file (auto speaker count)
    python run_diarization.py audio.wav

    # Hint an exact / bounded number of speakers
    python run_diarization.py audio.wav --num-speakers 2
    python run_diarization.py audio.wav --min-speakers 2 --max-speakers 5

    # Write an RTTM file next to the audio
    python run_diarization.py audio.wav --rttm out.rttm
"""

from __future__ import annotations

import argparse
import time

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

PIPELINE_ID = "pyannote/speaker-diarization-community-1"


def build_pipeline(token: str | None, device: str) -> Pipeline:
    pipeline = Pipeline.from_pretrained(PIPELINE_ID, token=token)
    if pipeline is None:
        raise SystemExit(
            f"Failed to load '{PIPELINE_ID}'. Accept the user conditions on "
            "Hugging Face and run `huggingface-cli login` (or pass --token)."
        )
    pipeline.to(torch.device(device))
    return pipeline


def sync_device(device: str) -> None:
    """Wait for async GPU work so timing is accurate."""
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="path to an audio file (wav, flac, ...)")
    parser.add_argument("--token", default=None, help="HF access token override")
    parser.add_argument(
        "--device", default="cpu", help="torch device: cpu, xpu, or cuda"
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

    pipeline = build_pipeline(args.token, args.device)

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
        #output = pipeline(args.audio, hook=hook)
    sync_device(args.device)
    elapsed = time.perf_counter() - start

    diarization = output.speaker_diarization

    print(f"\n# diarization for {args.audio} ({args.device}, {elapsed:.1f}s)")
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
