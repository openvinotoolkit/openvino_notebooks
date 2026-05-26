from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

NOTEBOOK_DIR = Path(__file__).resolve().parent
OPENVINO_NOTEBOOKS_DIR = NOTEBOOK_DIR.parents[1]
UTILS_DIR = OPENVINO_NOTEBOOKS_DIR / "utils"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from cmd_helper import optimum_cli


DEFAULT_REPRO_CONFIG: dict[str, Any] = {
    "model_id": "katuni4ka/tiny-random-stable-diffusion-3",
    "weight_format": "int4",
    "group_size": "-1",
    "ratio": "1.0",
    "load_t5": False,
    "height": 32,
    "width": 32,
    "steps": 4,
    "guidance_scale": 5.0,
    "seed": 141,
    "device": "GPU",
    "prompt": "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
}


def normalize_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_REPRO_CONFIG)
    if config:
        merged.update(config)
    return merged


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    python_path_parts = [str(NOTEBOOK_DIR), str(UTILS_DIR)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    return env


def cleanup_output_dir(output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)


def export_repro_model(output_dir: str | Path, config: dict[str, Any] | None = None, force_export: bool = False) -> Path:
    cfg = normalize_config(config)
    output_path = Path(output_dir)
    if force_export and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if any(output_path.iterdir()) and not force_export:
        return output_path

    additional_args = {
        "weight-format": cfg["weight_format"],
        "group-size": cfg["group_size"],
        "ratio": cfg["ratio"],
    }
    optimum_cli(cfg["model_id"], output_path, show_command=False, additional_args=additional_args)
    return output_path


def build_generate_only_script(output_dir: str | Path, config: dict[str, Any] | None = None) -> str:
    cfg = normalize_config(config)
    output_path = Path(output_dir).resolve()
    payload = json.dumps(cfg)
    return textwrap.dedent(
        f"""
        import json
        from pathlib import Path

        import openvino_genai as ov_genai
        from sd3_helper import init_pipeline_without_t5

        config = json.loads({payload!r})
        model_dir = Path({str(output_path)!r})

        if config[\"load_t5\"]:
            pipe = ov_genai.Text2ImagePipeline(model_dir, config[\"device\"])
        else:
            pipe = init_pipeline_without_t5(model_dir, config[\"device\"])

        image_tensor = pipe.generate(
            config[\"prompt\"],
            num_inference_steps=config[\"steps\"],
            guidance_scale=config[\"guidance_scale\"],
            height=config[\"height\"],
            width=config[\"width\"],
            generator=ov_genai.TorchGenerator(config[\"seed\"]),
        )

        print(
            json.dumps(
                {{
                    \"mode\": \"generate_only\",
                    \"model_dir\": str(model_dir),
                    \"shape\": list(image_tensor.data.shape),
                    \"device\": config[\"device\"],
                }}
            )
        )
        """
    )


def build_cold_repro_script(output_dir: str | Path, config: dict[str, Any] | None = None) -> str:
    cfg = normalize_config(config)
    output_path = Path(output_dir).resolve()
    payload = json.dumps(cfg)
    return textwrap.dedent(
        f"""
        import json
        import shutil
        from pathlib import Path

        import openvino_genai as ov_genai
        from cmd_helper import optimum_cli
        from sd3_helper import init_pipeline_without_t5

        config = json.loads({payload!r})
        model_dir = Path({str(output_path)!r})

        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        additional_args = {{
            \"weight-format\": config[\"weight_format\"],
            \"group-size\": config[\"group_size\"],
            \"ratio\": config[\"ratio\"],
        }}
        optimum_cli(config[\"model_id\"], model_dir, show_command=False, additional_args=additional_args)

        if config[\"load_t5\"]:
            pipe = ov_genai.Text2ImagePipeline(model_dir, config[\"device\"])
        else:
            pipe = init_pipeline_without_t5(model_dir, config[\"device\"])

        image_tensor = pipe.generate(
            config[\"prompt\"],
            num_inference_steps=config[\"steps\"],
            guidance_scale=config[\"guidance_scale\"],
            height=config[\"height\"],
            width=config[\"width\"],
            generator=ov_genai.TorchGenerator(config[\"seed\"]),
        )

        print(
            json.dumps(
                {{
                    \"mode\": \"cold_export_generate\",
                    \"model_dir\": str(model_dir),
                    \"shape\": list(image_tensor.data.shape),
                    \"device\": config[\"device\"],
                }}
            )
        )
        """
    )


def run_generate_only_once(output_dir: str | Path, config: dict[str, Any] | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-c", build_generate_only_script(output_dir, config=config)]
    return subprocess.run(command, capture_output=True, text=True, cwd=str(NOTEBOOK_DIR), env=_child_env())


def run_cold_repro_once(output_dir: str | Path, config: dict[str, Any] | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-c", build_cold_repro_script(output_dir, config=config)]
    return subprocess.run(command, capture_output=True, text=True, cwd=str(NOTEBOOK_DIR), env=_child_env())


def result_to_record(iteration: int, output_dir: str | Path, result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "output_dir": str(output_dir),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def print_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [item for item in results if item["returncode"] != 0]
    print(f"Completed {len(results)} runs, failures: {len(failures)}")
    for item in failures:
        print(f"--- iteration={item['iteration']} dir={item['output_dir']} rc={item['returncode']} ---")
        if item["stdout"]:
            print(item["stdout"][-4000:])
        if item["stderr"]:
            print(item["stderr"][-4000:])
    return failures


def reuse_model_generate_loop(
    iterations: int = 10,
    output_dir: str | Path = NOTEBOOK_DIR / "sd3_ci_repro" / "shared_model",
    force_export: bool = False,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = normalize_config(config)
    exported_dir = export_repro_model(output_dir, config=cfg, force_export=force_export)
    results = []
    for iteration in range(iterations):
        result = run_generate_only_once(exported_dir, config=cfg)
        results.append(result_to_record(iteration, exported_dir, result))
    print_failures(results)
    return results


def cold_export_generate_loop(
    iterations: int = 10,
    repro_root: str | Path = NOTEBOOK_DIR / "sd3_ci_repro",
    cleanup_passed_runs: bool = True,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = normalize_config(config)
    repro_root_path = Path(repro_root)
    repro_root_path.mkdir(parents=True, exist_ok=True)
    results = []
    for iteration in range(iterations):
        output_dir = repro_root_path / f"cold_run_{iteration:03d}_{uuid.uuid4().hex[:8]}"
        result = run_cold_repro_once(output_dir, config=cfg)
        record = result_to_record(iteration, output_dir, result)
        results.append(record)
        if cleanup_passed_runs and record["returncode"] == 0:
            cleanup_output_dir(output_dir)
    print_failures(results)
    return results


def cold_export_generate_task(
    iteration: int,
    repro_root: str | Path,
    cleanup_passed_runs: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = normalize_config(config)
    repro_root_path = Path(repro_root)
    repro_root_path.mkdir(parents=True, exist_ok=True)
    output_dir = repro_root_path / f"parallel_run_{iteration:03d}_{uuid.uuid4().hex[:8]}"
    result = run_cold_repro_once(output_dir, config=cfg)
    record = result_to_record(iteration, output_dir, result)
    if cleanup_passed_runs and record["returncode"] == 0:
        cleanup_output_dir(output_dir)
    return record


def parallel_cold_export_generate_loop(
    iterations: int = 8,
    workers: int = 4,
    repro_root: str | Path = NOTEBOOK_DIR / "sd3_ci_repro",
    cleanup_passed_runs: bool = True,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = normalize_config(config)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                cold_export_generate_task,
                iteration,
                repro_root,
                cleanup_passed_runs=cleanup_passed_runs,
                config=cfg,
            ): iteration
            for iteration in range(iterations)
        }
        for future in as_completed(future_map):
            record = future.result()
            results.append(record)
            print(f"iteration={record['iteration']} rc={record['returncode']} dir={record['output_dir']}")
    results.sort(key=lambda item: item["iteration"])
    print_failures(results)
    return results
