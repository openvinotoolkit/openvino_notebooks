import http.client
import io
import json
import re
import shutil
import subprocess
import urllib.error
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from nightly_report import build_report, github_api
from nightly_report.report_common import Warnings
from nightly_report.repository import get_last_modified
from nightly_report.workflow import os_family, parse_job_name, runner_label_to_os_device, runner_label_text, workflow_job_spec

SCHEMA_DIR = Path(build_report.__file__).resolve().parent / "schema"

RUN_ID = 555
LINUX = "cpu-ubuntu-22.04-py3.11"
WINDOWS = "cpu-windows-2022-py3.11"
IGPU = "igpu-ubuntu-22.04-py3.11"
DGPU = "dgpu-ubuntu-22.04-py3.11"

CELL_ERROR_LOG = """Executing treon version 0.1.4
Triggered test for test_b.ipynb
Start executing cell 0
Start executing cell 3
ERROR in testing test_b.ipynb

An error occurred while executing the following cell:
------------------
model = load("/tmp/xyz/model.xml")
------------------

\x1b[0;31m---------------------------------------------------------------------------\x1b[0m
\x1b[0;31mFileNotFoundError\x1b[0m                         Traceback (most recent call last)
Cell \x1b[0;32mIn[3], line 1\x1b[0m
\x1b[0;31mFileNotFoundError\x1b[0m: [Errno 2] No such file or directory: '/tmp/xyz/model.xml'

OpenVINO after notebook execution: 2026.4.1
"""

TIMEOUT_LOG = """Start executing cell 0
Start executing cell 7
[validate_notebooks] Timeout reached (1200s), process killed
"""

# Job log of job 101 (step 2 runs the notebooks; line numbers of the GitHub log viewer are given on the right).
JOB_101_LOG = [
    "Current runner version: '2.338.0'",
    "##[group]Run echo setup",
    "##[endgroup]",
    '##[group]Run IGNORE_ARGS=""',  # 1
    'IGNORE_ARGS=""',  # 2
    "##[endgroup]",  # 3
    "Testing notebook: /__w/repo/repo/notebooks/a/a.ipynb",  # 4
    "TEST DURATION [a.ipynb]: 1000.00 seconds",  # 5
    "Testing notebook: /__w/repo/repo/notebooks/b/b.ipynb",  # 6
    "Triggered test for test_b.ipynb",  # 7
    "Start executing cell 0",  # 8
    "\x1b[0;31mERROR in testing test_b.ipynb\x1b[0m",  # 9
    "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/xyz/model.xml'",  # 10
    "TEST DURATION [b.ipynb]: 60.00 seconds",  # 11
    "FAILED: ",  # 12
    "test_b.ipynb",
    "##[error]Process completed with exit code 1.",
]

WORKFLOW = """
jobs:
  collect_notebooks:
    runs-on: ubuntu-latest
  build_treon_linux:
    needs: [collect_notebooks, get_docker_tag]
    strategy:
      matrix:
        runs_on: [aks-linux-8-cores-32gb]
        python: ['3.11']
        batch: ['batch_0', 'batch_1']
    uses: ./.github/workflows/job_unix.yml
    with:
      runner: '["aks-linux-8-cores-32gb"]'
      python: ${{ matrix.python }}
      ignore_list: .ci/tensorflow.txt
  build_treon_windows:
    needs: collect_notebooks
    strategy:
      matrix:
        runs_on: [aks-win-16-cores-32gb-full]
        python: ['3.11']
        batch: ['batch_0', 'batch_1']
    uses: ./.github/workflows/job_windows.yml
    with:
      runs_on: ${{ matrix.runs_on }}
      python: ${{ matrix.python }}
      ignore_list: .ci/tensorflow.txt
  build_treon_igpu:
    needs: [collect_notebooks, get_docker_tag]
    strategy:
      matrix:
        python: ['3.11']
        batch: ['batch_0', 'batch_1']
    uses: ./.github/workflows/job_unix.yml
    with:
      runner: '["self-hosted", "igpu", "Linux"]'
      runner-group: 'Intel-GPU'
      python: ${{ matrix.python }}
      ignore_list: .ci/gpu_oom.txt .ci/not_allowed_resources.txt
  build_treon_dgpu:
    needs: [collect_notebooks, get_docker_tag]
    strategy:
      matrix:
        python: ['3.11']
        batch: ['batch_0', 'batch_1']
    uses: ./.github/workflows/job_unix.yml
    with:
      runner: '["self-hosted", "dgpu"]'
      runner-group: 'Intel-GPU'
      python: ${{ matrix.python }}
      ignore_list: .ci/not_allowed_resources.txt
  build_treon_gpu:
    if: ${{ false }}
    strategy:
      matrix:
        runs_on: ['gpu']
        python: ['3.11']
"""


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "-c", "user.name=Tester", "-c", "user.email=t@example.com", *args], text=True).strip()


def write_notebook(path: Path, title: str, tags: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"# {title} with [OpenVINO](https://openvino.ai)\n", "text"]}],
        "metadata": {"openvino_notebooks": {"tags": tags, "imageUrl": ""}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(content), encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for name in ("a", "b", "c", "d", "e", "tf"):
        write_notebook(root / "notebooks" / name / f"{name}.ipynb", f"Notebook {name.upper()}", {"categories": ["Demo"], "tasks": [name]})
    (root / ".ci").mkdir(parents=True)
    (root / ".ci" / "skipped_notebooks.yml").write_text(
        "- notebook: notebooks/e/e.ipynb\n  skips:\n    - os:\n        - windows-2022\n- notebook: notebooks/b/b.ipynb\n  skips:\n    - device:\n        - gpu\n",
        encoding="utf-8",
    )
    (root / ".ci" / "tensorflow.txt").write_text("notebooks/tf/tf.ipynb\n", encoding="utf-8")
    (root / ".ci" / "gpu_oom.txt").write_text("notebooks/c/c.ipynb\n", encoding="utf-8")
    (root / ".ci" / "not_allowed_resources.txt").write_text("notebooks/d/d.ipynb\nnotebooks/c/c.ipynb\n", encoding="utf-8")
    (root / ".ci" / "heavy_ubuntu_gpu.txt").write_text("notebooks/a/a.ipynb\n", encoding="utf-8")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "nightly.yml").write_text(WORKFLOW, encoding="utf-8")
    subprocess.check_call(["git", "init", "-q", str(root)])
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Initial commit")
    (root / "notebooks" / "a" / "README.md").write_text("changed", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Update notebook a")
    return root


def fragment(device_os: str, batch: str, job_id: int, planned: list[str], results: list[dict], complete: bool = True) -> dict:
    return {
        "schema_version": "1.0.0",
        "job": {
            "run_id": RUN_ID,
            "run_attempt": 1,
            "job_id": job_id,
            "batch": batch,
            "device": "cpu",
            "os": device_os,
            "python": "3.11",
            "python_full_version": "3.11.9",
            "platform": "test-platform",
            "runner_name": f"runner-{job_id}",
            "runner_label": "aks-linux-8-cores-32gb" if device_os.startswith("ubuntu") else "aks-win-16-cores-32gb-full",
            "container_image": "image:tag" if device_os.startswith("ubuntu") else None,
            "timeout_s": 1200,
            "separate_venv": True,
            "openvino_base_version": None,
            "started_at": "2026-10-08T00:46:00Z",
            "updated_at": "2026-10-08T02:00:00Z",
            "finished_at": "2026-10-08T02:00:00Z" if complete else None,
            "complete": complete,
        },
        "planned": planned,
        "results": results,
    }


def result(notebook: str, status: str, duration: float = 10.0, **extra) -> dict:
    data = {
        "notebook": notebook,
        "status": status,
        "skip_reason": None,
        "started_at": "2026-10-08T01:00:00Z",
        "finished_at": "2026-10-08T01:00:10Z",
        "duration_s": duration,
        "return_code": 0 if status == "passed" else 1,
        "peak_rss_bytes": 1024,
        "openvino_before": None,
        "openvino_after": "2026.4.1",
        "packages": {"openvino": "2026.4.1"},
        "env_id": "aaaaaaaaaaaa",
        "log_file": None,
        "error_hint": None,
    }
    data.update(extra)
    return data


def skipped(notebook: str, reason: str) -> dict:
    return {"notebook": notebook, "status": "skipped", "skip_reason": reason}


def write_fragment(root: Path, artifact: str, data: dict, logs: dict | None = None, envs: dict | None = None) -> None:
    directory = root / artifact
    (directory / "logs").mkdir(parents=True)
    (directory / "envs").mkdir()
    (directory / "fragment.json").write_text(json.dumps(data), encoding="utf-8")
    for name, text in (logs or {}).items():
        (directory / "logs" / name).write_text(text, encoding="utf-8")
    for env_id, text in (envs or {}).items():
        (directory / "envs" / f"{env_id}.txt").write_text(text, encoding="utf-8")


def api_job(job_id: int, name: str, conclusion: str, steps: list[tuple[str, str]], run_attempt: int = 1) -> dict:
    return {
        "id": job_id,
        "name": name,
        "conclusion": conclusion,
        "status": "completed",
        "run_attempt": run_attempt,
        "started_at": "2026-10-08T00:46:00Z",
        "completed_at": "2026-10-08T02:46:00Z",
        "runner_name": f"runner-{job_id}",
        "html_url": f"https://github.com/owner/repo/actions/runs/{RUN_ID}/job/{job_id}",
        "steps": [{"name": step, "conclusion": step_conclusion, "number": index + 1} for index, (step, step_conclusion) in enumerate(steps)],
    }


@pytest.fixture
def inputs(tmp_path: Path, repo: Path) -> dict:
    head_sha = git(repo, "rev-parse", "HEAD")
    plan = {
        "schema_version": "1.0.0",
        "seed": "1",
        "num_batches": 2,
        "notebooks_dir": "notebooks",
        "notebooks_total": 6,
        "batches": {
            "batch_0": ["notebooks/a/a.ipynb", "notebooks/b/b.ipynb", "notebooks/tf/tf.ipynb"],
            "batch_1": ["notebooks/c/c.ipynb", "notebooks/d/d.ipynb", "notebooks/e/e.ipynb"],
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    fragments = tmp_path / "fragments"
    write_fragment(
        fragments,
        "nightly-fragment-cpu-ubuntu-22.04-3.11-batch_0",
        fragment(
            "ubuntu-22.04",
            "batch_0",
            101,
            ["a/a.ipynb", "b/b.ipynb", "tf/tf.ipynb"],
            [
                skipped("tf/tf.ipynb", "ignore_list"),
                result("a/a.ipynb", "passed", 1000.0),
                result("b/b.ipynb", "failed", 60.0, log_file="logs/b__b.log", packages={"openvino": "2026.5.0.dev20261007"}),
            ],
        ),
        logs={"b__b.log": CELL_ERROR_LOG},
        envs={"aaaaaaaaaaaa": "openvino==2026.4.1\nnumpy==2.4.6\n"},
    )
    write_fragment(
        fragments,
        "nightly-fragment-cpu-ubuntu-22.04-3.11-batch_1",
        fragment(
            "ubuntu-22.04",
            "batch_1",
            102,
            ["c/c.ipynb", "d/d.ipynb", "e/e.ipynb"],
            [result("c/c.ipynb", "timeout", 1200.0, return_code=-42, log_file="logs/c__c.log"), result("d/d.ipynb", "passed", 30.0)],
            complete=False,
        ),
        logs={"c__c.log": TIMEOUT_LOG},
    )
    write_fragment(
        fragments,
        "nightly-fragment-cpu-windows-2022-3.11-batch_0",
        fragment(
            "windows-2022",
            "batch_0",
            201,
            ["a/a.ipynb", "b/b.ipynb", "tf/tf.ipynb"],
            [
                skipped("tf/tf.ipynb", "ignore_list"),
                result("a/a.ipynb", "passed", 50.0),
                result("b/b.ipynb", "error", None, return_code=None, error_hint="CalledProcessError: venv clone failed", env_id=None, packages={}),
            ],
        ),
    )

    linux_steps = [("Set env variables", "success"), ("Analysing with treon (Linux)", "success"), ("Analysing with treon (MacOS)", "skipped")]
    jobs = {
        "jobs": [
            api_job(1, "collect_notebooks", "success", []),
            api_job(2, "build_treon_gpu", "skipped", []),
            api_job(101, "build_treon_linux (aks-linux-8-cores-32gb, 3.11, batch_0) / build_treon", "success", linux_steps),
            api_job(102, "build_treon_linux (aks-linux-8-cores-32gb, 3.11, batch_1) / build_treon", "failure", linux_steps),
            api_job(201, "build_treon_windows (aks-win-16-cores-32gb-full, 3.11, batch_0) / build_treon", "success", [("Analysing with treon", "success")]),
            api_job(
                202,
                "build_treon_windows (aks-win-16-cores-32gb-full, 3.11, batch_1) / build_treon",
                "failure",
                [("Set env variables", "success"), ("Install python dependencies", "failure"), ("Analysing with treon", "skipped")],
            ),
            api_job(3, "build_dashboard_data", None, []),
        ]
    }
    artifacts = {
        "artifacts": [
            {"id": 9001, "name": "nightly-fragment-cpu-ubuntu-22.04-3.11-batch_0", "expired": False},
            {"id": 9002, "name": "nightly-fragment-cpu-ubuntu-22.04-3.11-batch_1", "expired": False},
        ]
    }
    run = {
        "id": RUN_ID,
        "name": "treon_nightly",
        "run_attempt": 1,
        "run_number": 88,
        "event": "schedule",
        "head_branch": "latest",
        "head_sha": head_sha,
        "created_at": "2026-10-08T00:43:15Z",
        "run_started_at": "2026-10-08T00:43:15Z",
        "html_url": f"https://github.com/owner/repo/actions/runs/{RUN_ID}",
        "head_commit": {"id": head_sha, "message": "Update notebook a\n\nbody", "timestamp": "2026-10-07T10:00:00Z", "author": {"name": "Tester"}},
    }
    paths = {}
    for name, data in (("jobs", jobs), ("artifacts", artifacts), ("run", run)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(data), encoding="utf-8")
    job_logs = tmp_path / "job_logs"
    job_logs.mkdir()
    (job_logs / "101.log").write_text(
        "\ufeff" + "".join(f"2026-10-08T01:00:{index:02d}.0000000Z {line}\r\n" for index, line in enumerate(JOB_101_LOG)), encoding="utf-8"
    )
    return {"repo": repo, "plan": plan_path, "fragments": fragments, "output": tmp_path / "out", "head_sha": head_sha, "job_logs": job_logs, **paths}


def build_args(inputs: dict, *extra: str) -> list[str]:
    return [
        "--fragments_dir",
        str(inputs["fragments"]),
        "--plan",
        str(inputs["plan"]),
        "--output_dir",
        str(inputs["output"]),
        "--repo_root",
        str(inputs["repo"]),
        "--ignore_list",
        ".ci/tensorflow.txt",
        "--repository",
        "owner/repo",
        "--run_id",
        str(RUN_ID),
        "--ov_branch",
        "master",
        "--docker_tag",
        "tag-1",
        "--workflow_file",
        ".github/workflows/nightly.yml",
        *extra,
    ]


def load_report(inputs: dict) -> dict:
    return json.loads((inputs["output"] / "nightly-report.json").read_text(encoding="utf-8"))


def by_key(report: dict) -> dict:
    return {(row["config_id"], row["notebook"]): row for row in report["results"]}


def test_full_report(inputs: dict, tmp_path: Path):
    summary_file = tmp_path / "step_summary.md"
    args = build_args(
        inputs,
        "--run_json",
        str(inputs["run"]),
        "--jobs_json",
        str(inputs["jobs"]),
        "--artifacts_json",
        str(inputs["artifacts"]),
        "--summary_file",
        str(summary_file),
        "--job_logs_dir",
        str(inputs["job_logs"]),
    )
    assert build_report.main(args) == 0

    report = load_report(inputs)
    validator = Draft202012Validator(json.loads((SCHEMA_DIR / "nightly-report.schema.json").read_text(encoding="utf-8")))
    assert not list(validator.iter_errors(report))

    run = report["run"]
    assert run["id"] == RUN_ID and run["number"] == 88 and run["event"] == "schedule" and run["branch"] == "latest"
    assert run["nightly_date"] == "2026-10-08"
    assert run["conclusion"] == "failure"
    assert run["finished_at"] == "2026-10-08T02:46:00Z" and run["wall_clock_s"] == pytest.approx(7365.0)
    assert run["commit"]["sha"] == inputs["head_sha"] and run["commit"]["message"] == "Update notebook a"
    assert run["inputs"] == {"ov_branch": "master", "docker_tag": "tag-1", "timeout_s": 1200, "num_batches": 2, "ignore_lists": [".ci/tensorflow.txt"]}

    assert [config["id"] for config in report["configs"]] == [LINUX, WINDOWS]
    linux_config = report["configs"][0]
    assert linux_config["os_family"] == "linux" and linux_config["container_image"] == "image:tag" and linux_config["job_ids"] == [101, 102]

    jobs = {job["id"]: job for job in report["jobs"]}
    assert set(jobs) == {101, 102, 201, 202}
    assert jobs[101]["treon_step_url"].endswith("/job/101#step:2:1")
    assert jobs[101]["logs_artifact_url"] == f"https://github.com/owner/repo/actions/runs/{RUN_ID}/artifacts/9001"
    assert jobs[101]["duration_s"] == pytest.approx(7200.0)
    assert jobs[202]["tests_started"] is False and jobs[202]["fragment_present"] is False
    assert jobs[202]["failed_step"]["name"] == "Install python dependencies" and jobs[202]["failed_step"]["url"].endswith("#step:2:1")

    rows = by_key(report)
    assert len(rows) == 12
    assert rows[(LINUX, "a/a.ipynb")]["status"] == "passed"
    assert rows[(LINUX, "tf/tf.ipynb")]["skip_reason"] == "ignore_list"
    assert rows[(LINUX, "e/e.ipynb")]["status"] == "not_run" and rows[(LINUX, "e/e.ipynb")]["not_run_reason"] == "job_interrupted"
    assert rows[(WINDOWS, "b/b.ipynb")]["status"] == "error"
    assert rows[(WINDOWS, "e/e.ipynb")]["status"] == "skipped" and rows[(WINDOWS, "e/e.ipynb")]["skip_reason"] == "skip_config"
    for notebook in ("c/c.ipynb", "d/d.ipynb"):
        assert rows[(WINDOWS, notebook)]["status"] == "not_run"
        assert rows[(WINDOWS, notebook)]["not_run_reason"] == "job_failed_before_tests:Install python dependencies"

    failures = {failure["id"]: failure for failure in report["failures"]}
    assert set(failures) == {f"{LINUX}/b/b.ipynb", f"{LINUX}/c/c.ipynb", f"{WINDOWS}/b/b.ipynb"}
    cell_error = failures[f"{LINUX}/b/b.ipynb"]
    assert cell_error["category"] == "cell_error" and cell_error["exception_type"] == "FileNotFoundError" and cell_error["cell_index"] == 3
    assert cell_error["links"]["log_file"] == "logs/b__b.log"
    assert cell_error["links"]["notebook_source"] == f"https://github.com/owner/repo/blob/{inputs['head_sha']}/notebooks/b/b.ipynb"
    # The piece of the job log around the failure and the link to it.
    job_101 = f"https://github.com/owner/repo/actions/runs/{RUN_ID}/job/101"
    assert cell_error["log_excerpt_source"] == "job_log" and cell_error["log_excerpt_lines"] == {"start": 7, "end": 10, "error_line": 9, "test_start": 7}
    assert cell_error["links"]["log_excerpt"] == cell_error["links"]["test_start"] == f"{job_101}#step:2:7"
    assert cell_error["log_excerpt"].splitlines() == [line.replace("\x1b[0;31m", "").replace("\x1b[0m", "") for line in JOB_101_LOG[9:13]]
    # Without a job log the piece comes from the captured notebook log and the link opens the notebook test step.
    timeout = failures[f"{LINUX}/c/c.ipynb"]
    assert timeout["log_excerpt_source"] == "fragment_log" and timeout["log_excerpt_lines"] is None
    assert "Timeout reached (1200s)" in timeout["log_excerpt"]
    assert timeout["links"]["log_excerpt"] == f"https://github.com/owner/repo/actions/runs/{RUN_ID}/job/102#step:2:1"
    assert failures[f"{LINUX}/c/c.ipynb"]["category"] == "timeout" and failures[f"{LINUX}/c/c.ipynb"]["cell_index"] == 7
    harness = failures[f"{WINDOWS}/b/b.ipynb"]
    assert harness["category"] == "harness_error" and harness["error_class"] == "harness" and harness["links"]["logs_artifact"] is None
    assert harness["log_excerpt"] is None and harness["log_excerpt_source"] is None
    assert harness["links"]["log_excerpt"] == f"https://github.com/owner/repo/actions/runs/{RUN_ID}/job/201#step:1:1"
    assert harness["links"]["test_start"] is None
    assert rows[(LINUX, "b/b.ipynb")]["failure_id"] == f"{LINUX}/b/b.ipynb"
    assert sum(group["occurrences"] for group in report["error_groups"]) == 3

    notebooks = {notebook["path"]: notebook for notebook in report["notebooks"]}
    assert set(notebooks) == {"a/a.ipynb", "b/b.ipynb", "c/c.ipynb", "d/d.ipynb", "e/e.ipynb", "tf/tf.ipynb"}
    assert notebooks["a/a.ipynb"]["title"] == "Notebook A with OpenVINO"
    assert notebooks["a/a.ipynb"]["tags"] == {"categories": ["Demo"], "tasks": ["a"]}
    assert notebooks["a/a.ipynb"]["dir_tree_sha"] == git(inputs["repo"], "rev-parse", "HEAD:notebooks/a")
    assert notebooks["a/a.ipynb"]["last_modified"]["message"] == "Update notebook a"
    assert notebooks["b/b.ipynb"]["last_modified"]["message"] == "Initial commit"

    summary = report["summary"]
    assert summary["results"]["planned"] == 12 and summary["results"]["executed"] == 6 and summary["results"]["not_run"] == 3
    assert summary["infra"]["jobs_failed_before_tests"] == 1

    environments = json.loads((inputs["output"] / "environments.json").read_text(encoding="utf-8"))
    assert environments == {"aaaaaaaaaaaa": {"openvino": "2026.4.1", "numpy": "2.4.6"}}
    assert (inputs["output"] / "nightly-report.schema.json").exists()
    markdown = (inputs["output"] / "summary.md").read_text(encoding="utf-8")
    assert markdown and markdown.strip() in summary_file.read_text(encoding="utf-8")
    assert not report["generator"]["warnings"]


def test_report_without_api(inputs: dict):
    assert build_report.main(build_args(inputs, "--no_api")) == 0
    report = load_report(inputs)
    assert {job["id"] for job in report["jobs"]} == {101, 102, 201}
    assert report["run"]["conclusion"] == "unknown"
    assert any("Job information is unavailable" in warning for warning in report["generator"]["warnings"])
    # Without job information, a fragment's missing notebooks are still reported.
    assert by_key(report)[(LINUX, "e/e.ipynb")]["status"] == "not_run"


def test_report_without_inputs(tmp_path: Path, repo: Path):
    output = tmp_path / "out"
    args = ["--fragments_dir", str(tmp_path / "missing"), "--output_dir", str(output), "--repo_root", str(repo), "--no_api", "--run_id", "1"]
    assert build_report.main(args) == 0
    report = json.loads((output / "nightly-report.json").read_text(encoding="utf-8"))
    assert report["results"] == [] and len(report["notebooks"]) == 6
    assert report["generator"]["warnings"]


def test_crashed_job_without_plan_is_reported_as_warning(inputs: dict):
    inputs["plan"].unlink()
    args = build_args(inputs, "--run_json", str(inputs["run"]), "--jobs_json", str(inputs["jobs"]), "--artifacts_json", str(inputs["artifacts"]))
    assert build_report.main(args) == 0
    report = load_report(inputs)
    assert not [row for row in report["results"] if row["job_id"] == 202]
    assert any("No fragment and no plan" in warning for warning in report["generator"]["warnings"])


def test_shallow_or_missing_git_is_tolerated(tmp_path: Path, inputs: dict):
    warnings = Warnings()
    assert get_last_modified(tmp_path, "notebooks", {"a"}, warnings) == {}
    assert warnings.items


def mocked_api_args(inputs: dict, jobs: list[dict], *extra: str) -> list[str]:
    inputs["jobs"].write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    return build_args(inputs, "--run_json", str(inputs["run"]), "--jobs_json", str(inputs["jobs"]), "--artifacts_json", str(inputs["artifacts"]), *extra)


def linux_skipped_before_start(inputs: dict) -> list[dict]:
    """The Linux matrix is skipped as a whole (get_docker_tag failed): GitHub reports one job without matrix values."""
    for directory in inputs["fragments"].glob("nightly-fragment-cpu-ubuntu-*"):
        shutil.rmtree(directory)
    jobs = json.loads(inputs["jobs"].read_text(encoding="utf-8"))["jobs"]
    jobs = [job for job in jobs if not job["name"].startswith("build_treon_linux")]
    return jobs + [api_job(4, "get_docker_tag", "failure", []), api_job(300, "build_treon_linux", "skipped", [])]


def test_jobs_skipped_before_matrix_expansion(inputs: dict):
    assert build_report.main(mocked_api_args(inputs, linux_skipped_before_start(inputs))) == 0
    report = load_report(inputs)
    rows = by_key(report)
    for notebook in ("a/a.ipynb", "b/b.ipynb", "c/c.ipynb", "d/d.ipynb", "e/e.ipynb"):
        assert (rows[(LINUX, notebook)]["status"], rows[(LINUX, notebook)]["not_run_reason"]) == ("not_run", "job_skipped")
    assert rows[(LINUX, "tf/tf.ipynb")]["skip_reason"] == "ignore_list"
    linux_jobs = [job for job in report["jobs"] if job["config_id"] == LINUX]
    assert [(job["id"], job["batch"], job["tests_started"]) for job in linux_jobs] == [(300, "batch_0", False), (300, "batch_1", False)]
    assert report["run"]["conclusion"] == "failure"
    assert report["summary"]["infra"]["jobs_failed_before_tests"] == 3
    assert not [warning for warning in report["generator"]["warnings"] if "matrix" in warning or "build_treon" in warning]
    assert not [warning for warning in report["generator"]["warnings"] if "Skip resolution" in warning]


def gpu_fragment(device: str, batch: str, job_id: int, planned: list[str], results: list[dict]) -> dict:
    data = fragment("ubuntu-22.04", batch, job_id, planned, results)
    labels = ["self-hosted", device, "Linux"] if device == "igpu" else ["self-hosted", device]
    data["job"].update(device=device, test_device="gpu", runner_label=json.dumps(labels), container_image="igpu-image:tag")
    return data


def test_igpu_and_dgpu_jobs(inputs: dict):
    """iGPU and dGPU jobs both validate for --device gpu but are separate configs; skips name their ignore list."""
    write_fragment(
        inputs["fragments"],
        "nightly-fragment-igpu-ubuntu-22.04-3.11-batch_0",
        gpu_fragment(
            "igpu",
            "batch_0",
            401,
            ["a/a.ipynb", "b/b.ipynb", "tf/tf.ipynb"],
            [
                {**skipped("a/a.ipynb", "ignore_list"), "skip_source": "heavy_ubuntu_gpu.txt"},
                skipped("b/b.ipynb", "skip_config"),
                result("tf/tf.ipynb", "failed", 5.0, log_file=None),
            ],
        ),
    )
    write_fragment(
        inputs["fragments"],
        "nightly-fragment-dgpu-ubuntu-22.04-3.11-batch_0",
        gpu_fragment("dgpu", "batch_0", 501, ["a/a.ipynb", "b/b.ipynb", "tf/tf.ipynb"], [result("tf/tf.ipynb", "passed", 5.0)]),
    )
    linux_steps = [("Set env variables", "success"), ("Analysing with treon (Linux)", "success")]
    jobs = json.loads(inputs["jobs"].read_text(encoding="utf-8"))["jobs"] + [
        api_job(401, "build_treon_igpu (3.11, batch_0) / build_treon", "failure", linux_steps),
        api_job(402, "build_treon_igpu (3.11, batch_1) / build_treon", "failure", [("Set env variables", "success"), ("Install GPU driver", "failure")]),
        api_job(501, "build_treon_dgpu (3.11, batch_0) / build_treon", "cancelled", linux_steps),
        api_job(502, "build_treon_dgpu (3.11, batch_1) / build_treon", "failure", [("GPU diagnostics (OpenVINO devices)", "failure")]),
    ]
    artifacts = json.loads(inputs["artifacts"].read_text(encoding="utf-8"))
    artifacts["artifacts"].append({"id": 9401, "name": "nightly-fragment-igpu-ubuntu-22.04-3.11-batch_0", "expired": False})
    inputs["artifacts"].write_text(json.dumps(artifacts), encoding="utf-8")
    assert build_report.main(mocked_api_args(inputs, jobs)) == 0

    report = load_report(inputs)
    validator = Draft202012Validator(json.loads((SCHEMA_DIR / "nightly-report.schema.json").read_text(encoding="utf-8")))
    assert not list(validator.iter_errors(report))
    configs = {config["id"]: config for config in report["configs"]}
    assert sorted(configs) == sorted([LINUX, WINDOWS, IGPU, DGPU])
    igpu, dgpu = configs[IGPU], configs[DGPU]
    assert (igpu["device"], igpu["test_device"], igpu["os_family"], igpu["runner_label"]) == ("igpu", "gpu", "linux", "self-hosted, igpu, Linux")
    assert igpu["ignore_lists"] == [".ci/gpu_oom.txt", ".ci/not_allowed_resources.txt", ".ci/heavy_ubuntu_gpu.txt"] and igpu["job_ids"] == [401, 402]
    assert (dgpu["device"], dgpu["runner_label"], dgpu["ignore_lists"]) == (
        "dgpu",
        "self-hosted, dgpu",
        [".ci/not_allowed_resources.txt", ".ci/heavy_ubuntu_gpu.txt"],
    )
    assert configs[LINUX]["test_device"] == "cpu" and configs[LINUX]["ignore_lists"] == [".ci/tensorflow.txt"]
    run_lists = report["run"]["inputs"]["ignore_lists"]
    assert run_lists[0] == ".ci/tensorflow.txt" and sorted(run_lists[1:]) == [".ci/gpu_oom.txt", ".ci/heavy_ubuntu_gpu.txt", ".ci/not_allowed_resources.txt"]

    rows = by_key(report)
    skip = lambda config, notebook: (rows[(config, notebook)]["status"], rows[(config, notebook)]["skip_reason"], rows[(config, notebook)]["skip_source"])
    # From fragments
    assert skip(IGPU, "a/a.ipynb") == ("skipped", "ignore_list", "heavy_ubuntu_gpu.txt")
    assert skip(IGPU, "b/b.ipynb") == ("skipped", "skip_config", None)
    assert rows[(IGPU, "tf/tf.ipynb")]["status"] == "failed" and rows[(DGPU, "tf/tf.ipynb")]["status"] == "passed"
    assert (rows[(DGPU, "a/a.ipynb")]["status"], rows[(DGPU, "a/a.ipynb")]["not_run_reason"]) == ("not_run", "job_cancelled")
    # Jobs failed before tests: skips resolved for --device gpu with the ignore lists of each job; the first listing file wins
    assert skip(IGPU, "c/c.ipynb") == ("skipped", "ignore_list", "gpu_oom.txt")
    assert skip(DGPU, "c/c.ipynb") == ("skipped", "ignore_list", "not_allowed_resources.txt")
    assert skip(IGPU, "d/d.ipynb") == ("skipped", "ignore_list", "not_allowed_resources.txt")
    assert (rows[(IGPU, "e/e.ipynb")]["status"], rows[(IGPU, "e/e.ipynb")]["not_run_reason"]) == ("not_run", "job_failed_before_tests:Install GPU driver")
    assert rows[(DGPU, "e/e.ipynb")]["not_run_reason"] == "job_failed_before_tests:GPU diagnostics (OpenVINO devices)"
    # CPU configs are not affected by the GPU-only skip config entry and the GPU ignore lists
    assert rows[(LINUX, "b/b.ipynb")]["status"] == "failed" and rows[(LINUX, "c/c.ipynb")]["status"] == "timeout"
    assert rows[(LINUX, "tf/tf.ipynb")]["skip_source"] is None

    igpu_jobs = {job["id"]: job for job in report["jobs"] if job["config_id"] == IGPU}
    assert igpu_jobs[401]["logs_artifact_url"].endswith("/artifacts/9401") and igpu_jobs[401]["fragment_present"]
    assert not igpu_jobs[402]["tests_started"]
    assert not [warning for warning in report["generator"]["warnings"] if "build_treon" in warning or "Skip resolution" in warning]
    assert IGPU in (inputs["output"] / "summary.md").read_text(encoding="utf-8")


def test_run_conclusion_ignores_jobs_running_in_parallel(inputs: dict):
    for directory in inputs["fragments"].iterdir():
        shutil.rmtree(directory)
    jobs = [
        api_job(1, "collect_notebooks", "success", []),
        api_job(4, "get_docker_tag", "cancelled", []),
        api_job(300, "build_treon_linux", "skipped", []),
        api_job(301, "build_treon_windows", "skipped", []),
        api_job(5, "aggregate_notebooks_reports", "failure", []),
    ]
    assert build_report.main(mocked_api_args(inputs, jobs)) == 0
    report = load_report(inputs)
    assert report["run"]["conclusion"] == "cancelled"
    assert len(report["jobs"]) == 4 and {config["id"]: config["job_ids"] for config in report["configs"]} == {LINUX: [300], WINDOWS: [301]}
    assert report["summary"]["infra"]["infra_failure_rate"] == 1.0


def test_unexpanded_job_without_workflow_file_is_reported_as_warning(inputs: dict):
    assert build_report.main(mocked_api_args(inputs, linux_skipped_before_start(inputs), "--workflow_file", "")) == 0
    report = load_report(inputs)
    assert not [row for row in report["results"] if row["config_id"] == LINUX]
    assert any("has no expanded matrix" in warning for warning in report["generator"]["warnings"])


def test_stale_fragment_from_previous_attempt_is_ignored(inputs: dict):
    jobs = json.loads(inputs["jobs"].read_text(encoding="utf-8"))["jobs"]
    jobs = [job for job in jobs if job["id"] != 102] + [
        api_job(
            402,
            "build_treon_linux (aks-linux-8-cores-32gb, 3.11, batch_1) / build_treon",
            "failure",
            [("Install required packages (container)", "failure"), ("Analysing with treon (Linux)", "skipped")],
            run_attempt=2,
        )
    ]
    assert build_report.main(mocked_api_args(inputs, jobs)) == 0
    report = load_report(inputs)
    rows = by_key(report)
    for notebook in ("c/c.ipynb", "d/d.ipynb", "e/e.ipynb"):
        assert rows[(LINUX, notebook)]["job_id"] == 402
        assert rows[(LINUX, notebook)]["not_run_reason"] == "job_failed_before_tests:Install required packages (container)"
    assert {job["id"]: job["fragment_present"] for job in report["jobs"]}[402] is False
    assert any("not from the latest job 402" in warning for warning in report["generator"]["warnings"])


def test_github_api_errors_only_warn(monkeypatch):
    warnings = Warnings()
    api = github_api.GitHubApi("https://api.example.com", "owner/repo", "token", warnings)
    calls = []

    def disconnected(*args, **kwargs):
        calls.append(1)
        raise http.client.RemoteDisconnected("Remote end closed connection without response")

    monkeypatch.setattr(github_api.urllib.request, "urlopen", disconnected)
    monkeypatch.setattr(github_api.time, "sleep", lambda seconds: None)
    assert api.get_run(1) is None
    assert len(calls) == 3 and len(warnings.items) == 1 and "Remote end closed connection" in warnings.items[0]

    def not_found(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://api.example.com", 404, "Not Found", {}, None)

    calls.clear()
    monkeypatch.setattr(github_api.urllib.request, "urlopen", not_found)
    assert api.get_jobs(1) is None and len(calls) == 1


def test_workflow_matrices_of_the_nightly_workflow():
    repo_root = Path(build_report.__file__).resolve().parents[2]
    args = build_report.parse_arguments(
        ["--fragments_dir", "x", "--output_dir", "y", "--repo_root", str(repo_root), "--workflow_file", ".github/workflows/treon_nightly.yml", "--no_api"]
    )
    builder = build_report.ReportBuilder(args)
    matrices = builder._workflow_matrices()
    expected = {
        "build_treon_linux": ("ubuntu-22.04", "cpu", "aks-linux-8-cores-32gb"),
        "build_treon_windows": ("windows-2022", "cpu", "aks-win-16-cores-32gb-full"),
        "build_treon_igpu": ("ubuntu-22.04", "igpu", "self-hosted, igpu, Linux"),
        "build_treon_dgpu": ("ubuntu-22.04", "dgpu", "self-hosted, dgpu"),
    }
    assert set(expected) <= set(matrices)
    for job_key, (os_name, device, runner_label) in expected.items():
        assert matrices[job_key]
        for combination in matrices[job_key]:
            assert (combination["os"], combination["device"], combination["runner_label"]) == (os_name, device, runner_label)
            assert re.fullmatch(r"\d+\.\d+", combination["python"]) and re.fullmatch(r"batch_\d+", combination["batch"])
    # Every test job uploads a dashboard fragment and the report job waits for all of them
    workflow = yaml.safe_load((repo_root / ".github/workflows/treon_nightly.yml").read_text(encoding="utf-8"))
    assert all(workflow["jobs"][job_key]["with"].get("dashboard_data") is True for job_key in expected)
    assert set(expected) <= set(workflow["jobs"]["build_dashboard_data"]["needs"])
    igpu_lists = builder._ignore_lists("build_treon_igpu", "igpu", "ubuntu-22.04", "3.11")
    assert igpu_lists[-1] == ".ci/heavy_ubuntu_gpu.txt" and ".ci/gpu_oom.txt" in igpu_lists
    assert builder._ignore_lists("build_treon_linux", "cpu", "ubuntu-22.04", "3.11") == [".ci/tensorflow.txt"]
    windows_lists = builder._ignore_lists("build_treon_windows", "cpu", "windows-2022", "3.11")
    assert windows_lists == [".ci/tensorflow.txt"]
    windows_py313_lists = builder._ignore_lists("build_treon_windows", "cpu", "windows-2022", "3.13")
    assert windows_py313_lists == [*windows_lists, ".ci/win_py313_flaky.txt"]
    for item in igpu_lists + builder._ignore_lists("build_treon_dgpu", "dgpu", "ubuntu-22.04", "3.11") + windows_py313_lists:
        assert (repo_root / item).is_file(), item


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "build_treon_linux (aks-linux-8-cores-32gb, 3.11, batch_2) / build_treon",
            {"runner_label": "aks-linux-8-cores-32gb", "python": "3.11", "batch": "batch_2", "os": "ubuntu-22.04", "device": "cpu"},
        ),
        (
            "build_treon_windows (aks-win-16-cores-32gb-full, 3.13, batch_0) / build_treon",
            {"runner_label": "aks-win-16-cores-32gb-full", "python": "3.13", "batch": "batch_0", "os": "windows-2022", "device": "cpu"},
        ),
        (
            "build_treon_gpu (gpu, 3.12, ubuntu:22.04) / build_treon",
            {"runner_label": "gpu", "python": "3.12", "batch": None, "os": "ubuntu-22.04", "device": "gpu"},
        ),
        (
            "build_treon_igpu (3.11, batch_4) / build_treon",
            {"runner_label": None, "python": "3.11", "batch": "batch_4", "os": "ubuntu-22.04", "device": "igpu"},
        ),
        ("build_treon_gpu", None),
        ("collect_notebooks", None),
        ("aggregate_notebooks_reports (x, 3.11) / y", None),
    ],
)
def test_parse_job_name(name, expected):
    assert parse_job_name(name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "build_treon_linux (aks-linux-8-cores-32gb, 3.11, batch_1) / build_treon",
            {"runner_label": "aks-linux-8-cores-32gb", "python": "3.11", "batch": "batch_1", "os": "ubuntu-22.04", "device": "cpu"},
        ),
        (
            "build_treon_windows (aks-win-16-cores-32gb-full, 3.11, batch_0) / build_treon",
            {"runner_label": "aks-win-16-cores-32gb-full", "python": "3.11", "batch": "batch_0", "os": "windows-2022", "device": "cpu"},
        ),
        (
            "build_treon_igpu (3.11, batch_1) / build_treon",
            {"runner_label": "self-hosted, igpu, Linux", "python": "3.11", "batch": "batch_1", "os": "ubuntu-22.04", "device": "igpu"},
        ),
        (
            "build_treon_dgpu (3.11, batch_0) / build_treon",
            {"runner_label": "self-hosted, dgpu", "python": "3.11", "batch": "batch_0", "os": "ubuntu-22.04", "device": "dgpu"},
        ),
    ],
)
def test_parse_job_name_with_workflow(name, expected):
    specs = {key: workflow_job_spec(job) for key, job in yaml.safe_load(WORKFLOW)["jobs"].items() if key.startswith("build_treon")}
    assert parse_job_name(name, specs) == expected


def test_runner_mapping_and_os_family():
    assert runner_label_to_os_device("aks-linux-8-cores-32gb") == ("ubuntu-22.04", "cpu")
    assert runner_label_to_os_device("macos-latest") == ("macos-13", "cpu")
    assert runner_label_to_os_device("aks-win-16-cores-32gb-full") == ("windows-2022", "cpu")
    assert runner_label_to_os_device("self-hosted, igpu, Linux") == ("ubuntu-22.04", "igpu")
    assert runner_label_to_os_device("self-hosted, dgpu") == ("ubuntu-22.04", "dgpu")
    assert runner_label_to_os_device("runner", "./.github/workflows/job_windows.yml") == ("windows-2022", "cpu")
    assert runner_label_to_os_device("windows-looking-label", "./.github/workflows/job_unix.yml") == ("ubuntu-22.04", "cpu")
    assert runner_label_text('["self-hosted", "igpu", "Linux"]') == "self-hosted, igpu, Linux"
    assert runner_label_text("aks-linux-8-cores-32gb") == "aks-linux-8-cores-32gb" and runner_label_text("") is None and runner_label_text("[x") == "[x"
    assert os_family("ubuntu-22.04") == "linux" and os_family("windows-2022") == "windows" and os_family("macos-13") == "macos" and os_family("x") == "unknown"


def test_job_log_download_does_not_forward_the_token(monkeypatch):
    requests = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, timeout):
        requests.append(request)
        return Response(b"2026-10-08T01:00:00.0000000Z line\n" * 10)

    monkeypatch.setattr(github_api.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(github_api, "JOB_LOG_MAX_BYTES", 100)
    api = github_api.GitHubApi("https://api.example.com", "owner/repo", "secret", Warnings())
    text = api.get_job_log(7)
    assert text == ("2026-10-08T01:00:00.0000000Z line\n" * 10)[:100]
    request = requests[0]
    assert request.full_url == "https://api.example.com/repos/owner/repo/actions/jobs/7/logs"
    # Unredirected headers are not sent to the storage URL the API redirects to.
    assert request.unredirected_hdrs.get("Authorization") == "Bearer secret" and "Authorization" not in request.headers
