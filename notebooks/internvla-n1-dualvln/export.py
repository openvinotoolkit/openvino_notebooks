#!/usr/bin/env python3
"""Export InternVLA-N1 components to accuracy-first OpenVINO IRs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import torch
from PIL import Image
from optimum.exporters.openvino import export_from_model
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, AutoTokenizer

from internvla.checkpoint import load_qwen_model, read_checkpoint_config
from internvla.system1 import load_system1_context_encoder, load_system1_denoiser
from internvla.system2 import (
    TRAJECTORY_QUERY_COUNT,
    TRAJECTORY_TOKEN_ID,
    add_trajectory_latent_output,
    load_latent_queries,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "pretrained"
DEFAULT_OUTPUT_DIR = ROOT / "openvino_model"
COMPONENTS = ("s1_denoiser", "s1_context_encoder", "qwen2_5_vl")
WEIGHT_PRECISION = "f16"
ARGS: argparse.Namespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy()


def comparison(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    delta = actual64 - reference64
    denominator = max(float(np.linalg.norm(reference64.ravel())), 1e-12)
    cosine_denominator = max(
        float(np.linalg.norm(reference64.ravel()) * np.linalg.norm(actual64.ravel())),
        1e-12,
    )
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "relative_l2_error": float(np.linalg.norm(delta.ravel()) / denominator),
        "cosine_similarity": float(np.dot(reference64.ravel(), actual64.ravel()) / cosine_denominator),
    }


def convert_save_validate(
    name: str,
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    output_dir: Path,
    validate: bool,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        reference_outputs = model(*example_inputs)
    if isinstance(reference_outputs, torch.Tensor):
        reference_outputs = (reference_outputs,)

    traced_model = torch.jit.trace(model, example_inputs, check_trace=False, strict=False)
    ov_model = ov.convert_model(traced_model)
    if len(ov_model.inputs) != len(input_names):
        raise RuntimeError(f"{name}: converted {len(ov_model.inputs)} inputs, expected {len(input_names)}")
    if len(ov_model.outputs) != len(output_names):
        raise RuntimeError(f"{name}: converted {len(ov_model.outputs)} outputs, expected {len(output_names)}")
    for port, port_name in zip(ov_model.inputs, input_names, strict=True):
        port.get_tensor().set_names({port_name})
    for port, port_name in zip(ov_model.outputs, output_names, strict=True):
        port.get_tensor().set_names({port_name})

    xml_path = output_dir / f"{name}.xml"
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)
    result: dict[str, Any] = {
        "xml": xml_path.name,
        "weight_precision": WEIGHT_PRECISION,
        "io_precision": "f32",
        "inputs": {port_name: list(example.shape) for port_name, example in zip(input_names, example_inputs, strict=True)},
        "outputs": {port_name: list(reference.shape) for port_name, reference in zip(output_names, reference_outputs, strict=True)},
        "validation": None,
    }
    if validate:
        compiled = ov.Core().compile_model(xml_path, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        ov_outputs = compiled({port_name: as_numpy(example) for port_name, example in zip(input_names, example_inputs, strict=True)})
        metrics = {}
        for index, (port_name, reference) in enumerate(zip(output_names, reference_outputs, strict=True)):
            metrics[port_name] = comparison(as_numpy(reference), np.asarray(ov_outputs[compiled.output(index)]))
        result["validation"] = {"device": "CPU", "precision": "f32", "metrics": metrics}
        print(f"{name}: {json.dumps(metrics, sort_keys=True)}")
    return result


def export_s1_denoiser(output_dir: Path, validate: bool) -> dict[str, Any]:
    torch.manual_seed(0)
    model = load_system1_denoiser(ARGS.model_dir)
    inputs = (
        torch.randn(2, 32, 3),
        torch.tensor([1000, 1000], dtype=torch.int64),
        torch.randn(2, 36, 768),
    )
    return convert_save_validate(
        "s1_denoiser",
        model,
        inputs,
        ("trajectories", "timestep", "context"),
        ("noise_prediction",),
        output_dir,
        validate,
    )


def export_s1_context_encoder(output_dir: Path, validate: bool) -> dict[str, Any]:
    torch.manual_seed(0)
    model = load_system1_context_encoder(ARGS.model_dir)
    inputs = (torch.rand(1, 2, 3, 224, 224), torch.randn(1, 4, 3584))
    return convert_save_validate(
        "s1_context_encoder",
        model,
        inputs,
        ("images", "trajectory_latents"),
        ("context",),
        output_dir,
        validate,
    )


def prepare_validation_inputs(processor: Any) -> dict[str, torch.Tensor]:
    pixels = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "You are a navigation assistant. Move forward."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return dict(processor(text=[text], images=[image], return_tensors="pt"))


def prepare_trajectory_latent_inputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    model_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    latent_queries = torch.from_numpy(load_latent_queries(model_dir))

    with torch.no_grad():
        text_embeddings = model.model.language_model.embed_tokens(input_ids)
        image_embeddings = model.model.visual(pixel_values.to(model.model.visual.dtype), grid_thw=image_grid_thw)
        image_mask = input_ids == model.config.image_token_id
        if int(image_mask.sum()) != image_embeddings.shape[0]:
            raise RuntimeError("trajectory prefill image-token mismatch: " f"tokens={int(image_mask.sum())}, features={image_embeddings.shape[0]}")
        text_embeddings[image_mask] = image_embeddings.to(text_embeddings.dtype)
        inputs_embeds = torch.cat((text_embeddings, latent_queries.to(text_embeddings.dtype)), dim=1)
        trajectory_ids = torch.full(
            (input_ids.shape[0], TRAJECTORY_QUERY_COUNT),
            TRAJECTORY_TOKEN_ID,
            dtype=input_ids.dtype,
        )
        position_ids, _ = model.model.get_rope_index(torch.cat((input_ids, trajectory_ids), dim=1), image_grid_thw)

        captured: list[torch.Tensor] = []

        def capture_post_norm(_module: torch.nn.Module, _arguments: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured.append(output.detach())

        hook = model.model.language_model.norm.register_forward_hook(capture_post_norm)
        try:
            model.model.language_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
        finally:
            hook.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one final-norm output, captured {len(captured)}")
    reference = as_numpy(captured[0][:, -TRAJECTORY_QUERY_COUNT:, :])
    ov_inputs = {
        "inputs_embeds": as_numpy(inputs_embeds),
        "attention_mask": np.ones(inputs_embeds.shape[:2], dtype=np.int64),
        "position_ids": as_numpy(position_ids),
        "beam_idx": np.zeros(input_ids.shape[0], dtype=np.int32),
    }
    return ov_inputs, reference, as_numpy(latent_queries)


def augment_qwen_language_model(output_dir: Path) -> None:
    xml_path = output_dir / "openvino_language_model.xml"
    model = ov.Core().read_model(xml_path)
    add_trajectory_latent_output(model)
    temporary_dir = output_dir / ".system2_augmentation"
    temporary_dir.mkdir(exist_ok=True)
    temporary_xml = temporary_dir / xml_path.name
    ov.save_model(model, temporary_xml, compress_to_fp16=True)
    del model
    gc.collect()
    os.replace(temporary_xml, xml_path)
    os.replace(temporary_xml.with_suffix(".bin"), xml_path.with_suffix(".bin"))
    temporary_dir.rmdir()


def compress_qwen_models_to_fp16(output_dir: Path) -> None:
    temporary_dir = output_dir / ".fp16_compression"
    temporary_dir.mkdir(exist_ok=True)
    for xml_path in sorted(output_dir.glob("openvino_*.xml")):
        if xml_path.name == "openvino_language_model.xml":
            continue
        model = ov.Core().read_model(xml_path)
        temporary_xml = temporary_dir / xml_path.name
        ov.save_model(model, temporary_xml, compress_to_fp16=True)
        del model
        gc.collect()
        os.replace(temporary_xml, xml_path)
        os.replace(temporary_xml.with_suffix(".bin"), xml_path.with_suffix(".bin"))
    temporary_dir.rmdir()


def validate_trajectory_latents(
    output_dir: Path,
    inputs: dict[str, np.ndarray],
    reference: np.ndarray,
) -> dict[str, Any]:
    compiled = ov.Core().compile_model(
        output_dir / "openvino_language_model.xml",
        "CPU",
        {"INFERENCE_PRECISION_HINT": "f32"},
    )
    request = compiled.create_infer_request()
    request.reset_state()
    outputs = request.infer(inputs)
    actual = np.asarray(outputs[compiled.output("trajectory_latents")])
    metrics = comparison(reference, actual)
    if actual.shape != reference.shape or metrics["cosine_similarity"] < 0.9999:
        raise RuntimeError("System 2 trajectory-latent replay failed: " f"actual_shape={actual.shape}, reference_shape={reference.shape}, metrics={metrics}")
    print(f"qwen2_5_vl trajectory latents: {json.dumps(metrics, sort_keys=True)}")
    return {
        "device": "CPU",
        "precision": "f32",
        "shape": list(actual.shape),
        "metrics": metrics,
    }


def export_qwen(output_dir: Path, validate: bool) -> dict[str, Any]:
    processor = AutoProcessor.from_pretrained(ARGS.model_dir, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(ARGS.model_dir)
    model = load_qwen_model(ARGS.model_dir, dtype=torch.float32)
    if validate:
        inputs = prepare_validation_inputs(processor)
        with torch.no_grad():
            reference_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        expected = as_numpy(reference_ids)
        latent_inputs, latent_reference, latent_queries = prepare_trajectory_latent_inputs(
            model, reference_ids, inputs["pixel_values"], inputs["image_grid_thw"], ARGS.model_dir
        )
    else:
        latent_queries = load_latent_queries(ARGS.model_dir)
    export_from_model(
        model,
        output_dir,
        task="image-text-to-text",
        stateful=True,
        preprocessors=[processor, tokenizer],
    )
    processor.save_pretrained(output_dir)
    augment_qwen_language_model(output_dir)
    compress_qwen_models_to_fp16(output_dir)
    np.save(output_dir / "latent_queries.npy", latent_queries.astype(np.float16))
    del model
    gc.collect()
    artifacts = sorted(path.name for path in output_dir.glob("openvino_*.xml"))
    if not artifacts:
        raise RuntimeError("Optimum Intel produced no Qwen OpenVINO IR files")
    result: dict[str, Any] = {
        "exporter": "optimum-intel",
        "task": "image-text-to-text",
        "stateful": True,
        "weight_precision": WEIGHT_PRECISION,
        "io_precision": "f32",
        "artifacts": artifacts,
        "trajectory_latents": {
            "boundary": "post_final_rmsnorm",
            "query_artifact": "latent_queries.npy",
            "query_precision": "f16",
            "output": "trajectory_latents",
            "shape": [1, TRAJECTORY_QUERY_COUNT, 3584],
            "validation": None,
        },
        "validation": None,
    }
    if validate:
        ov_model = OVModelForVisualCausalLM.from_pretrained(
            output_dir,
            device="CPU",
            ov_config={"INFERENCE_PRECISION_HINT": "f32"},
        )
        generated = ov_model.generate(**inputs, max_new_tokens=32, do_sample=False)
        actual = generated.detach().cpu().numpy()
        if not np.array_equal(actual, expected):
            mismatch = np.argwhere(actual != expected) if actual.shape == expected.shape else np.empty((0, 2))
            raise RuntimeError(
                "Qwen OpenVINO token replay failed: "
                f"actual_shape={actual.shape}, expected_shape={expected.shape}, "
                f"first_mismatch={mismatch[0].tolist() if mismatch.size else None}"
            )
        result["validation"] = {
            "device": "CPU",
            "precision": "f32",
            "reference": "checkpoint_float32",
            "input_source": "synthetic_seed_0",
            "scope": "conversion_smoke_test",
            "generated_ids_exact": True,
            "generated_shape": list(actual.shape),
        }
        result["trajectory_latents"]["validation"] = validate_trajectory_latents(output_dir, latent_inputs, latent_reference)
        print("qwen2_5_vl: generated token IDs match the float32 checkpoint reference exactly")
    return result


def main() -> None:
    global ARGS
    ARGS = parse_args()
    ARGS.model_dir = ARGS.model_dir.resolve()
    ARGS.output_dir = ARGS.output_dir.resolve()
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)

    raw_config = read_checkpoint_config(ARGS.model_dir)
    if raw_config.get("hidden_size") != 3584 or raw_config.get("vocab_size") != 152064:
        raise ValueError("checkpoint dimensions do not match InternVLA-N1")
    validate = not ARGS.skip_validation
    exporters = {
        "s1_denoiser": lambda: export_s1_denoiser(ARGS.output_dir, validate),
        "s1_context_encoder": lambda: export_s1_context_encoder(ARGS.output_dir, validate),
        "qwen2_5_vl": lambda: export_qwen(ARGS.output_dir, validate),
    }
    config_path = ARGS.output_dir / "export_manifest.json"
    records: dict[str, Any] = {}
    if config_path.exists():
        existing_manifest = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_manifest.get("weight_precision") != WEIGHT_PRECISION:
            raise ValueError("Use a new output directory when changing quantization")
        records.update(existing_manifest.get("components", {}))
    for component in COMPONENTS:
        if component not in ARGS.components:
            continue
        print(f"Exporting {component}...")
        records[component] = exporters[component]()
        gc.collect()

    manifest = {
        "schema_version": 1,
        "checkpoint_index_sha256": sha256(ARGS.model_dir / "model.safetensors.index.json"),
        "checkpoint": str(ARGS.model_dir),
        "versions": {
            "torch": torch.__version__,
            "openvino": ov.__version__,
            "transformers": package_version("transformers"),
            "diffusers": package_version("diffusers"),
            "timm": package_version("timm"),
            "optimum_intel": package_version("optimum-intel"),
            "nncf": package_version("nncf"),
        },
        "weight_precision": WEIGHT_PRECISION,
        "io_precision": "f32",
        "components": records,
    }
    config_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved export manifest to {config_path}")


if __name__ == "__main__":
    main()
