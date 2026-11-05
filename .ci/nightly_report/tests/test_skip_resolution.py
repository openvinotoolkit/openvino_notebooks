from pathlib import Path

import pytest
import yaml

import validate_notebooks
from nightly_report.constants import SkipReason
from validate_notebooks import NotebookStatus, get_ignore_list_notebooks, get_ignored_notebooks_from_yaml, prepare_test_plan, resolve_skip_reasons

CI_DIR = Path(__file__).resolve().parents[2]
NOT_ALLOWED_NOTEBOOKS = {
    Path("notebooks/aloha-act/aloha-act.ipynb"),
    Path("notebooks/detectron2-to-openvino/detectron2-to-openvino.ipynb"),
    Path("notebooks/gdrnpp-full-pipeline/gdrnpp-full-pipeline.ipynb"),
    Path("notebooks/multimodal-rag/multimodal-rag-llamaindex.ipynb"),
    Path("notebooks/tensorflow-object-detection-to-openvino/tensorflow-object-detection-api-with-openvino.ipynb"),
}

SKIP_CONFIG = """\
- notebook: notebooks/os-skip/os-skip.ipynb
  skips:
    - os:
        - ubuntu-22.04
        - windows-2022
- notebook: notebooks/python-skip/python-skip.ipynb
  skips:
    - python:
        - '3.12'
- notebook: notebooks/device-skip/device-skip.ipynb
  skips:
    - device:
        - gpu
- notebook: notebooks/multi-skip/multi-skip.ipynb
  skips:
    - os:
        - macos-13
    - device:
        - npu
- notebook: notebooks/both/both.ipynb
  skips:
    - os:
        - ubuntu-22.04
"""


@pytest.fixture
def skip_config(tmp_path) -> Path:
    path = tmp_path / "skipped_notebooks.yml"
    path.write_text(SKIP_CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def ignore_txt(tmp_path) -> Path:
    path = tmp_path / "ignore.txt"
    path.write_text("notebooks/txt-ignored/txt-ignored.ipynb\n\n   \n  notebooks/both/both.ipynb  \n", encoding="utf-8")
    return path


def config(os_name=None, python=None, device=None) -> dict:
    return {"os": os_name, "python": python, "device": device}


@pytest.mark.parametrize(
    "validation_config, expected",
    [
        (config("ubuntu-22.04", "3.11", "cpu"), {"notebooks/os-skip/os-skip.ipynb", "notebooks/both/both.ipynb"}),
        (config("windows-2022", "3.12", "cpu"), {"notebooks/os-skip/os-skip.ipynb", "notebooks/python-skip/python-skip.ipynb"}),
        (config("macos-13", "3.13", "gpu"), {"notebooks/device-skip/device-skip.ipynb", "notebooks/multi-skip/multi-skip.ipynb"}),
        (config("windows-2022", "3.11", "npu"), {"notebooks/os-skip/os-skip.ipynb", "notebooks/multi-skip/multi-skip.ipynb"}),
    ],
)
def test_get_ignored_notebooks_from_yaml(skip_config, validation_config, expected):
    notebooks = get_ignored_notebooks_from_yaml(validation_config, skip_config)
    assert all(isinstance(n, Path) for n in notebooks)
    assert len(notebooks) == len(set(notebooks))
    assert {n.as_posix() for n in notebooks} == expected


def test_get_skip_config_notebooks_missing_file(tmp_path, capsys):
    missing = tmp_path / "missing.yml"
    assert get_ignored_notebooks_from_yaml(config("ubuntu-22.04", "3.11", "cpu"), missing) == []
    assert f"Skipped notebooks config yaml file does not exist at path '{missing}'." in capsys.readouterr().out


def test_get_skip_config_notebooks_warns_about_missing_config_values(skip_config, capsys):
    notebooks = get_ignored_notebooks_from_yaml(config("ubuntu-22.04", None, None), skip_config)
    assert {n.as_posix() for n in notebooks} == {"notebooks/os-skip/os-skip.ipynb", "notebooks/both/both.ipynb"}
    out = capsys.readouterr().out
    assert "Warning: validation config argument 'python' is not provided." in out
    assert "Warning: validation config argument 'device' is not provided." in out


def test_get_ignore_list_notebooks(ignore_txt):
    assert get_ignore_list_notebooks(None) == []
    notebooks = get_ignore_list_notebooks([str(ignore_txt), "notebooks/direct/direct.ipynb"])
    assert notebooks == [
        Path("notebooks/txt-ignored/txt-ignored.ipynb"),
        Path("notebooks/both/both.ipynb"),
        Path("notebooks/direct/direct.ipynb"),
    ]


@pytest.mark.parametrize("job_name", ["build_treon_igpu", "build_treon_dgpu"])
def test_nightly_gpu_jobs_skip_not_allowed_resources(job_name, skip_config):
    repo_root = CI_DIR.parent
    workflow = yaml.safe_load((repo_root / ".github/workflows/treon_nightly.yml").read_text(encoding="utf-8"))
    ignore_lists = workflow["jobs"][job_name]["with"]["ignore_list"].split()
    assert ".ci/not_allowed_resources.txt" in ignore_lists
    if job_name == "build_treon_igpu":
        assert {".ci/gpu_oom.txt", ".ci/igpu_timeouts.txt"} <= set(ignore_lists)
    assert NOT_ALLOWED_NOTEBOOKS <= set(get_ignore_list_notebooks([str(CI_DIR / "not_allowed_resources.txt")]))

    test_plan = prepare_test_plan(
        config("ubuntu-22.04", "3.11", "gpu"),
        sorted(path.as_posix() for path in NOT_ALLOWED_NOTEBOOKS),
        str(skip_config),
        [str(repo_root / path) for path in ignore_lists],
        repo_root / "notebooks",
    )
    for path in NOT_ALLOWED_NOTEBOOKS:
        report = test_plan[path.relative_to("notebooks")]
        assert report["status"] == NotebookStatus.SKIPPED
        assert report["skip_reason"] == SkipReason.IGNORE_LIST


def test_not_allowed_resources_list_only_used_by_nightly_gpu_jobs():
    workflow = yaml.safe_load((CI_DIR.parent / ".github/workflows/treon_nightly.yml").read_text(encoding="utf-8"))
    jobs = {name for name, job in workflow["jobs"].items() if ".ci/not_allowed_resources.txt" in job.get("with", {}).get("ignore_list", "").split()}
    assert jobs == {"build_treon_igpu", "build_treon_dgpu"}


def test_resolve_skip_reasons_precedence(skip_config, ignore_txt):
    reasons = resolve_skip_reasons(config("ubuntu-22.04", "3.11", "cpu"), skip_config, [str(ignore_txt), "notebooks/direct/direct.ipynb"])
    assert reasons == {
        "notebooks/os-skip/os-skip.ipynb": SkipReason.SKIP_CONFIG,
        "notebooks/both/both.ipynb": SkipReason.SKIP_CONFIG,
        "notebooks/txt-ignored/txt-ignored.ipynb": SkipReason.IGNORE_LIST,
        "notebooks/direct/direct.ipynb": SkipReason.IGNORE_LIST,
    }


def test_resolve_skip_reasons_without_ignore_list(skip_config):
    reasons = resolve_skip_reasons(config("macos-13", "3.12", "cpu"), skip_config, None)
    assert reasons == {
        "notebooks/python-skip/python-skip.ipynb": SkipReason.SKIP_CONFIG,
        "notebooks/multi-skip/multi-skip.ipynb": SkipReason.SKIP_CONFIG,
    }


def test_resolve_skip_reasons_on_repository_configs():
    reasons = resolve_skip_reasons(config("ubuntu-22.04", "3.11", "cpu"), CI_DIR / "skipped_notebooks.yml", [str(CI_DIR / "tensorflow.txt")])
    assert reasons
    assert set(reasons.values()) <= set(SkipReason.ALL)
    assert all(path.startswith("notebooks/") and path.endswith(".ipynb") for path in reasons)


def test_skip_reason_constants_match_dashboard_contract():
    assert validate_notebooks.SKIP_CONFIG == SkipReason.SKIP_CONFIG
    assert validate_notebooks.IGNORE_LIST == SkipReason.IGNORE_LIST
    assert set(SkipReason.ALL) == {validate_notebooks.SKIP_CONFIG, validate_notebooks.IGNORE_LIST}


def test_validate_notebooks_imports_nightly_report_lazily():
    # The Docker image ships validate_notebooks.py without the nightly_report package
    source = (CI_DIR / "validate_notebooks.py").read_text(encoding="utf-8")
    assert not any(line.startswith(("import nightly_report", "from nightly_report")) for line in source.splitlines())
