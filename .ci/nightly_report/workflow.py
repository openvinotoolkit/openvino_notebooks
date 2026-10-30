"""Maps the test jobs of the treon_nightly workflow to test configurations: job names, runner labels, matrices and ignore lists.

Mirrors the workflow files: the 'Set env variables' steps of job_unix.yml / job_windows.yml and the test job definitions of
treon_nightly.yml.
"""

import itertools
import json
import re
from pathlib import Path
from typing import Iterable, Optional

from nightly_report.report_common import Warnings

TEST_JOB_PREFIX = "build_treon"
JOB_MATRIX_RE = re.compile(r"^(?P<prefix>[\w-]+)\s*\((?P<matrix>[^)]*)\)")
MATRIX_EXPRESSION_RE = re.compile(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}")
PYTHON_RE = re.compile(r"\d+\.\d+")
BATCH_RE = re.compile(r"batch_\d+")
UNIX_REUSABLE_WORKFLOW = "job_unix.yml"
WINDOWS_REUSABLE_WORKFLOW = "job_windows.yml"
GPU_DEVICE_TAGS = ("igpu", "dgpu", "gpu")  # most specific first
# job_unix.yml adds this ignore list to the job's ones on GPU
UNIX_GPU_IGNORE_LIST = ".ci/heavy_ubuntu_gpu.txt"
# job_windows.yml adds this ignore list to the job's ones on Python 3.13
WINDOWS_PY313_IGNORE_LIST = ".ci/win_py313_flaky.txt"


def os_family(os_name: str) -> str:
    lowered = os_name.lower()
    for prefix, family in (("ubuntu", "linux"), ("linux", "linux"), ("windows", "windows"), ("macos", "macos")):
        if lowered.startswith(prefix):
            return family
    return "unknown"


def runner_label_text(runner) -> Optional[str]:
    """Human-readable runner label(s): '["self-hosted", "igpu", "Linux"]' -> 'self-hosted, igpu, Linux'."""
    if runner is None:
        return None
    text = str(runner).strip()
    if text.startswith("["):
        try:
            labels = json.loads(text)
        except ValueError:
            labels = None
        if isinstance(labels, list):
            text = ", ".join(str(label) for label in labels)
    return text or None


def runner_label_to_os_device(runner_label: str, uses: str = "") -> tuple[str, str]:
    """Mirror of the 'Set env variables' steps of job_unix.yml (cpu / igpu / dgpu / gpu device tags) and job_windows.yml.

    `uses` is the reusable workflow called by the job; without it the OS is guessed from the labels."""
    lowered = runner_label.lower()
    device = next((tag for tag in GPU_DEVICE_TAGS if tag in lowered), "cpu")
    if uses.endswith(WINDOWS_REUSABLE_WORKFLOW) or (not uses.endswith(UNIX_REUSABLE_WORKFLOW) and "win" in lowered):
        os_name = "windows-2022"
    elif "mac" in lowered:
        os_name = "macos-13"
    else:
        os_name = "ubuntu-22.04"
    return os_name, device


def validation_device(device: str) -> str:
    """Device passed to validate_notebooks.py --device (skip config matching) for a dashboard device tag."""
    return "gpu" if device in GPU_DEVICE_TAGS else device


def workflow_job_spec(job: dict) -> dict:
    """The parts of a test job definition of the workflow needed to map its matrix jobs to configs."""
    inputs = job.get("with") or {}
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    matrix = {key: values for key, values in matrix.items() if key not in ("include", "exclude") and isinstance(values, list)}
    runner = inputs.get("runner") if inputs.get("runner") is not None else inputs.get("runs_on")
    return {
        "uses": str(job.get("uses") or ""),
        "runner": str(runner) if runner is not None else None,
        "matrix": matrix,
        "ignore_list": str(inputs.get("ignore_list") or "").split(),
        "run_all_notebooks": inputs.get("run_all_notebooks") is True,
    }


def job_combination(job_key: str, python: str, batch: Optional[str], runner: Optional[str], spec: Optional[dict] = None) -> dict:
    label = runner_label_text(runner)
    # The job id takes part in the guess, e.g. 'build_treon_igpu' or 'build_treon_windows'
    os_name, device = runner_label_to_os_device(f"{label or ''} {job_key}", (spec or {}).get("uses", ""))
    return {"runner_label": label, "python": python, "batch": batch, "os": os_name, "device": device}


def resolve_runner(spec: dict, matrix: dict) -> Optional[str]:
    if spec["runner"] is None:
        return matrix.get("runs_on")
    return MATRIX_EXPRESSION_RE.sub(lambda match: str(matrix.get(match.group(1), "")), spec["runner"])


def parse_job_name(name: str, specs: Optional[dict[str, dict]] = None) -> Optional[dict]:
    """'build_treon_linux (aks-linux-8-cores-32gb, 3.11, batch_2) / build_treon' -> matrix values.

    `specs` (job id -> workflow_job_spec) map the values to matrix keys; without them they are recognized by their format."""
    match = JOB_MATRIX_RE.match(name)
    if not match or not match.group("prefix").startswith(TEST_JOB_PREFIX):
        return None
    job_key = match.group("prefix")
    values = [value.strip() for value in match.group("matrix").split(",") if value.strip()]
    if not values:
        return None
    spec = (specs or {}).get(job_key)
    if spec and len(spec["matrix"]) == len(values):
        matrix = dict(zip(spec["matrix"], values))
        python, batch, runner = matrix.get("python"), matrix.get("batch"), resolve_runner(spec, matrix)
    else:
        python = next((value for value in values if PYTHON_RE.fullmatch(value)), None)
        batch = next((value for value in values if BATCH_RE.fullmatch(value)), None)
        runner = next((value for value in values if value not in (python, batch)), None)
    if python is None or not PYTHON_RE.fullmatch(python):
        return None
    return job_combination(job_key, python, batch, runner, spec)


def load_workflow(repo_root: Path, workflow_file: Optional[str], warnings: Warnings) -> tuple[dict[str, Optional[dict]], set[str]]:
    """Reads the workflow file: test job id -> workflow_job_spec with its expanded matrix under 'combinations'
    (None for jobs disabled with a constant false `if`) and the ids of the jobs the test jobs depend on (`needs`)."""
    specs: dict[str, Optional[dict]] = {}
    upstream: set[str] = set()
    if not workflow_file:
        return specs, upstream
    try:
        import yaml

        workflow = yaml.safe_load((repo_root / workflow_file).read_text(encoding="utf-8")) or {}
    except Exception as error:
        warnings.add(f"Cannot read workflow file '{workflow_file}': {error}")
        return specs, upstream
    for job_key, job in (workflow.get("jobs") or {}).items():
        if not str(job_key).startswith(TEST_JOB_PREFIX) or not isinstance(job, dict):
            continue
        if str(job.get("if", "")).replace(" ", "") in ("False", "false", "${{false}}"):
            specs[job_key] = None
            continue
        needs = job.get("needs") or []
        upstream.update([needs] if isinstance(needs, str) else needs)
        spec = workflow_job_spec(job)
        combinations = []
        if "python" in spec["matrix"]:
            keys = list(spec["matrix"])
            for values in itertools.product(*(spec["matrix"][key] for key in keys)):
                matrix = {key: str(value) for key, value in zip(keys, values)}
                combinations.append(job_combination(job_key, matrix["python"], matrix.get("batch"), resolve_runner(spec, matrix), spec))
        spec["combinations"] = combinations
        specs[job_key] = spec
    return specs, upstream


def job_ignore_lists(specs: dict[str, Optional[dict]], job_key: Optional[str], device: str, os_name: str, python: str, default: Iterable[str]) -> list[str]:
    """Ignore lists passed to validate_notebooks.py by a test job: from its workflow definition (found by job id or by
    config), else `default`."""
    specs = {key: spec for key, spec in specs.items() if spec}
    spec = specs.get(job_key or "")
    if spec is None:
        spec = next(
            (
                candidate
                for candidate in specs.values()
                if any((c["device"], c["os"], c["python"]) == (device, os_name, python) for c in candidate["combinations"])
            ),
            None,
        )
    if spec is None:
        return list(default)
    if spec.get("run_all_notebooks"):
        return []
    ignore_lists = list(spec["ignore_list"])
    if spec["uses"].endswith(UNIX_REUSABLE_WORKFLOW) and validation_device(device) == "gpu" and os_family(os_name) == "linux":
        ignore_lists.append(UNIX_GPU_IGNORE_LIST)
    if spec["uses"].endswith(WINDOWS_REUSABLE_WORKFLOW) and python == "3.13":
        ignore_lists.append(WINDOWS_PY313_IGNORE_LIST)
    return ignore_lists
