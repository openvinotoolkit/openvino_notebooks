from pathlib import Path

import pytest

import validate_notebooks
from nightly_report.constants import SkipReason
from split_notebooks import collect_notebooks
from validate_notebooks import get_ignore_list_notebooks, get_ignored_notebooks_from_yaml, resolve_skip_reasons

CI_DIR = Path(__file__).resolve().parents[2]

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


@pytest.mark.parametrize(
    "os_name, python, device",
    [
        ("ubuntu-22.04", "3.11", "cpu"),
        ("ubuntu-22.04", "3.12", "cpu"),
        ("ubuntu-22.04", "3.13", "cpu"),
        ("windows-2022", "3.11", "cpu"),
        ("windows-2022", "3.12", "cpu"),
        ("windows-2022", "3.13", "cpu"),
        ("ubuntu-22.04", "3.11", "gpu"),
    ],
)
def test_no_skips_config_runs_all_collected_notebooks(os_name, python, device):
    repo_root = CI_DIR.parent
    notebooks = collect_notebooks(repo_root / "notebooks")
    assert notebooks
    plan = validate_notebooks.prepare_test_plan(
        config(os_name, python, device),
        notebooks,
        "no_skips.yml",
        None,
        repo_root / "notebooks",
    )
    selected = [report for report in plan.values() if report["status"] != validate_notebooks.NotebookStatus.SKIPPED]
    assert {report["path"].as_posix() for report in selected} == set(notebooks)
    assert all(report["skip_reason"] is None for report in selected)


def test_skip_reason_constants_match_dashboard_contract():
    assert validate_notebooks.SKIP_CONFIG == SkipReason.SKIP_CONFIG
    assert validate_notebooks.IGNORE_LIST == SkipReason.IGNORE_LIST
    assert set(SkipReason.ALL) == {validate_notebooks.SKIP_CONFIG, validate_notebooks.IGNORE_LIST}


def test_validate_notebooks_imports_nightly_report_lazily():
    # The Docker image ships validate_notebooks.py without the nightly_report package
    source = (CI_DIR / "validate_notebooks.py").read_text(encoding="utf-8")
    assert not any(line.startswith(("import nightly_report", "from nightly_report")) for line in source.splitlines())
