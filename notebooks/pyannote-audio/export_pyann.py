# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: MIT

"""Export the pyannote diarization pipeline's neural blocks to OpenVINO IR.

The ``community-1`` speaker-diarization pipeline contains two heavy neural
blocks. This script converts each to OpenVINO IR so ``run_diarization_ov.py``
can load them directly (no conversion at inference time):

    * segmentation model (PyanNet)            -> ov_models/segmentation.xml/.bin
    * embedding ResNet   (WeSpeakerResNet34)  -> ov_models/embedding_resnet.xml/.bin

The embedding's fbank front end (Kaldi + torch.vmap) is NOT traceable, so only
the ResNet part is exported; fbank stays in PyTorch at inference time.

The pipeline is gated -- accept the conditions once on Hugging Face and log in
with ``huggingface-cli login`` (or pass ``--token``).

Examples:
    python export_pyann.py                 # export both to ./ov_models
    python export_pyann.py --force         # re-export even if IR exists
    python export_pyann.py --output-dir /tmp/ir
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import torch
import openvino as ov
from pyannote.audio import Pipeline

PIPELINE_ID = "pyannote/speaker-diarization-community-1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "ov_models"

SEGMENTATION_XML = "segmentation.xml"
EMBEDDING_XML = "embedding_resnet.xml"


def _prepare_for_trace(model: torch.nn.Module) -> None:
    """Make a pyannote (Lightning) model traceable.

    pyannote models are PyTorch Lightning modules whose ``trainer`` property
    raises during TorchScript's export scan, so attach a dummy trainer.
    """
    model.eval()
    for module in model.modules():
        module._trainer = MagicMock()


def export_segmentation(pipeline: Pipeline, xml_path: Path) -> None:
    """Convert the whole segmentation model to OV IR (dynamic batch)."""
    inference = pipeline._segmentation
    model = inference.model
    _prepare_for_trace(model)

    n = int(inference.duration * model.audio.sample_rate)
    example = torch.randn(1, 1, n)

    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False, check_trace=False)
    ov_model = ov.convert_model(
        traced, example_input=example, input=[ov.PartialShape([-1, 1, n])]
    )
    ov.save_model(ov_model, xml_path)
    print(f"segmentation -> {xml_path}  (input [-1, 1, {n}])")


def export_embedding_resnet(pipeline: Pipeline, xml_path: Path) -> None:
    """Convert the embedding ResNet to OV IR (fbank stays in PyTorch)."""
    model = pipeline._embedding.model_
    _prepare_for_trace(model)

    # probe fbank feature dimension (num_mel_bins)
    with torch.inference_mode():
        fbank = model.compute_fbank(torch.randn(2, 1, 32000))
    num_frames, num_mels = fbank.shape[1], fbank.shape[2]

    class ResnetWrap(torch.nn.Module):
        def __init__(self, resnet: torch.nn.Module):
            super().__init__()
            self.resnet = resnet

        def forward(self, fbank, weights):
            return self.resnet(fbank, weights=weights)[1]

    wrap = ResnetWrap(model.resnet).eval()
    example = (torch.randn(2, num_frames, num_mels), torch.ones(2, 50))

    with torch.inference_mode():
        traced = torch.jit.trace(wrap, example, strict=False, check_trace=False)
    ov_model = ov.convert_model(
        traced,
        example_input=example,
        input=[ov.PartialShape([-1, -1, num_mels]), ov.PartialShape([-1, -1])],
    )
    ov.save_model(ov_model, xml_path)
    print(f"embedding ResNet -> {xml_path}  (fbank [-1, -1, {num_mels}], weights [-1, -1])")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=None, help="HF access token override")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="folder to write the IR into (default: ./ov_models)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-export even if IR already exists"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_xml = output_dir / SEGMENTATION_XML
    emb_xml = output_dir / EMBEDDING_XML

    if not args.force and seg_xml.exists() and emb_xml.exists():
        print(f"IR already present in {output_dir} (use --force to re-export).")
        return

    pipeline = Pipeline.from_pretrained(PIPELINE_ID, token=args.token)
    if pipeline is None:
        raise SystemExit(
            f"Failed to load '{PIPELINE_ID}'. Accept the user conditions on "
            "Hugging Face and run `huggingface-cli login` (or pass --token)."
        )
    pipeline.to(torch.device("cpu"))

    export_segmentation(pipeline, seg_xml)
    export_embedding_resnet(pipeline, emb_xml)
    print(f"\nDone. IR written to {output_dir}")


if __name__ == "__main__":
    main()
