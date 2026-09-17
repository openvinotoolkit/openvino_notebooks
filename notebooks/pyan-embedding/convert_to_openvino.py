# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: MIT

"""Convert pyannote/embedding (XVectorSincNet) to OpenVINO IR format.

Usage:
    python convert_to_openvino.py [--output-dir models] [--static-seconds N]

By default the time axis is dynamic so the IR accepts any waveform length.
Pass ``--static-seconds 3`` to bake in a fixed-length input instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import openvino as ov
import torch
from pyannote.audio import Model

from common import MODEL_ID, SAMPLE_RATE


def _neutralize_trainer(model: torch.nn.Module) -> None:
    """Stop Lightning's ``trainer`` property from raising during the export scan.

    pyannote models are PyTorch Lightning modules. When TorchScript scans for
    exported methods it calls ``hasattr`` on every attribute, and the
    ``trainer`` property raises ``RuntimeError`` when no Trainer is attached,
    which aborts tracing. Assigning a dummy trainer makes the getter succeed.
    """
    for module in model.modules():
        try:
            module._trainer = MagicMock()
        except Exception:
            pass


class EmbeddingWrapper(torch.nn.Module):
    """Plain nn.Module wrapper exposing a single clean ``forward``."""

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(waveform)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models", help="where to write the IR")
    parser.add_argument(
        "--static-seconds",
        type=float,
        default=None,
        help="bake a fixed input length (seconds) instead of a dynamic time axis",
    )
    parser.add_argument(
        "--use-auth-token",
        default=None,
        help="HF token (defaults to the cached huggingface-cli login token)",
    )
    args = parser.parse_args()

    print(f"Loading {MODEL_ID} ...")
    model = Model.from_pretrained(MODEL_ID, use_auth_token=args.use_auth_token).eval()
    _neutralize_trainer(model)
    wrapped = EmbeddingWrapper(model).eval()

    # Trace with a 2 s example; the resulting IR can stay dynamic on the time axis.
    example_samples = int(SAMPLE_RATE * 2.0)
    example_input = torch.randn(1, 1, example_samples)

    if args.static_seconds is not None:
        time_dim = int(SAMPLE_RATE * args.static_seconds)
        input_shape = ov.PartialShape([1, 1, time_dim])
        example_input = torch.randn(1, 1, time_dim)
        print(f"Converting with STATIC input shape {list(input_shape)} ...")
    else:
        input_shape = ov.PartialShape([1, 1, -1])  # dynamic time axis
        print(f"Converting with DYNAMIC input shape {list(input_shape)} ...")

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_input, strict=False)
        ov_model = ov.convert_model(
            traced,
            example_input=example_input,
            input=[input_shape],
        )

    ov_model.inputs[0].get_node().set_friendly_name("waveform")
    ov_model.outputs[0].get_node().set_friendly_name("embedding")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "pyannote_embedding.xml"
    ov.save_model(ov_model, str(xml_path))

    print(f"Saved IR to: {xml_path}")
    print(f"          and {xml_path.with_suffix('.bin')}")


if __name__ == "__main__":
    main()
