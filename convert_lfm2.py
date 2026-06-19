"""
Convert LiquidAI LFM2 models to OpenVINO IR with INT4 weight compression.

Usage:
    python convert_lfm2.py --model_id LiquidAI/LFM2-8B-A1B --output LFM2-8B-A1B-int4-ov
    python convert_lfm2.py --model_id LiquidAI/LFM2-24B-A2B --output LFM2-24B-A2B-int4-ov

Requirements:
    pip install git+https://github.com/huggingface/optimum-intel.git nncf openvino
"""

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = {
    "LFM2-8B-A1B": {
        "model_id": "LiquidAI/LFM2-8B-A1B",
        "output": "LFM2-8B-A1B-int4-ov",
    },
    "LFM2-24B-A2B": {
        "model_id": "LiquidAI/LFM2-24B-A2B",
        "output": "LFM2-24B-A2B-int4-ov",
    },
}


def get_export_command(model_id: str, output_dir: str, group_size: int = 128, ratio: float = 0.8, sym: bool = False) -> str:
    command = (
        f"optimum-cli export openvino"
        f" --model {model_id}"
        f" --task text-generation-with-past"
        f" --weight-format int4"
        f" --group-size {group_size}"
        f" --ratio {ratio}"
        f" --trust-remote-code"
    )
    if sym:
        command += " --sym"
    command += f" {output_dir}"
    return command


def main():
    parser = argparse.ArgumentParser(description="Convert LFM2 models to OpenVINO IR with INT4 compression")
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="HuggingFace model ID (e.g. LiquidAI/LFM2-8B-A1B). If not specified, converts all models.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Defaults to <model_name>-int4-ov",
    )
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size (default: 128)")
    parser.add_argument("--ratio", type=float, default=0.8, help="Compression ratio (default: 0.8)")
    parser.add_argument("--sym", action="store_true", help="Use symmetric quantization")
    args = parser.parse_args()

    if args.model_id:
        models_to_convert = [
            {
                "model_id": args.model_id,
                "output": args.output or (args.model_id.split("/")[-1] + "-int4-ov"),
            }
        ]
    else:
        models_to_convert = list(MODELS.values())

    for model_info in models_to_convert:
        model_id = model_info["model_id"]
        output_dir = Path(model_info["output"])

        if (output_dir / "openvino_model.xml").exists():
            print(f"✅ Model already converted: {output_dir}")
            continue

        command = get_export_command(
            model_id=model_id,
            output_dir=str(output_dir),
            group_size=args.group_size,
            ratio=args.ratio,
            sym=args.sym,
        )

        print(f"⌛ Converting {model_id} to INT4 OpenVINO IR...")
        print(f"Command: {command}")
        print()

        result = subprocess.run(command.split(), check=False, capture_output=False)

        if result.returncode != 0:
            print(f"❌ Conversion failed for {model_id} (exit code {result.returncode})")
            sys.exit(1)

        print(f"✅ Successfully converted {model_id} → {output_dir}")
        print()


if __name__ == "__main__":
    main()
