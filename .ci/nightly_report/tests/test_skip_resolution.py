from pathlib import Path

import pytest

import skip_resolution
import validate_notebooks
from nightly_report.constants import SkipReason
from skip_resolution import get_ignore_list_notebooks, get_skip_config_notebooks, resolve_skip_reasons

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
def test_get_skip_config_notebooks(skip_config, validation_config, expected):
    notebooks = get_skip_config_notebooks(validation_config, skip_config)
    assert all(isinstance(n, Path) for n in notebooks)
    assert len(notebooks) == len(set(notebooks))
    assert {n.as_posix() for n in notebooks} == expected


def test_get_skip_config_notebooks_missing_file(tmp_path, capsys):
    missing = tmp_path / "missing.yml"
    assert get_skip_config_notebooks(config("ubuntu-22.04", "3.11", "cpu"), missing) == []
    assert f"Skipped notebooks config yaml file does not exist at path '{missing}'." in capsys.readouterr().out


def test_get_skip_config_notebooks_warns_about_missing_config_values(skip_config, capsys):
    notebooks = get_skip_config_notebooks(config("ubuntu-22.04", None, None), skip_config)
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


def test_validate_notebooks_keeps_backward_compatible_name():
    assert validate_notebooks.get_ignored_notebooks_from_yaml is skip_resolution.get_skip_config_notebooks


def test_skip_reason_constants_match_dashboard_contract():
    assert skip_resolution.SKIP_CONFIG == SkipReason.SKIP_CONFIG
    assert skip_resolution.IGNORE_LIST == SkipReason.IGNORE_LIST
    assert set(SkipReason.ALL) == {skip_resolution.SKIP_CONFIG, skip_resolution.IGNORE_LIST}


def test_skip_resolution_does_not_depend_on_nightly_report():
    source = (CI_DIR / "skip_resolution.py").read_text(encoding="utf-8")
    assert "import nightly_report" not in source and "from nightly_report" not in source
