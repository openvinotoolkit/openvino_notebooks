import csv
import json
import os
import shutil
import subprocess  # nosec B404 - runs the validation script under test
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import validate_notebooks
from nightly_report import fragment as fragment_module
from nightly_report.constants import SkipReason
from nightly_report.fragment import env_id_for, parse_pip_freeze
from validate_notebooks import NotebookStatus, prepare_test_plan, run_subprocess_with_timeout

CI_DIR = Path(__file__).resolve().parents[2]
FRAGMENT_SCHEMA = json.loads((CI_DIR / "nightly_report" / "schema" / "fragment.schema.json").read_text(encoding="utf-8"))
TIMEOUT_MARKER = "[validate_notebooks] Timeout reached (2s), process killed"

ALLOCATE_AND_PRINT = "import time; data = b'x' * (64 * 1024 * 1024); print('hello'); print('world', flush=True); time.sleep(1.5)"

NOTEBOOKS = ["pass", "fail", "slow", "missing", "crash", "harness", "skipcfg", "ignored", "outofscope"]
TESTING_LIST = [n for n in NOTEBOOKS if n != "outofscope"]

SKIP_CONFIG = """\
- notebook: notebooks/skipcfg/skipcfg.ipynb
  skips:
    - os:
        - ubuntu-22.04
- notebook: notebooks/outofscope/outofscope.ipynb
  skips:
    - os:
        - ubuntu-22.04
- notebook: notebooks/other-os/other-os.ipynb
  skips:
    - os:
        - windows-2022
"""

FREEZE = "openvino==2026.1.0\nnumpy==2.2.6\nrequests==2.33.0\n"


def assert_valid_fragment(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator(FRAGMENT_SCHEMA, format_checker=Draft202012Validator.FORMAT_CHECKER).validate(data)
    return data


@pytest.fixture
def restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


# --- run_subprocess_with_timeout ---------------------------------------------------------------------------------------


def test_run_subprocess_writes_log_and_peak_rss(tmp_path, capsys):
    log_file = tmp_path / "logs" / "nb.log"
    stats = {}
    retcode, duration = run_subprocess_with_timeout([sys.executable, "-c", ALLOCATE_AND_PRINT], 60, log_file=log_file, stats=stats)
    assert retcode == 0
    assert duration > 1
    assert log_file.read_text(encoding="utf-8") == "hello\nworld\n"
    assert stats["peak_rss_bytes"] > 64 * 1024 * 1024
    assert "harness_error" not in stats
    out = capsys.readouterr().out
    assert "hello\n" in out and "world\n" in out


def test_run_subprocess_appends_to_log(tmp_path):
    log_file = tmp_path / "nb.log"
    log_file.write_text("previous\n", encoding="utf-8")
    retcode, _ = run_subprocess_with_timeout([sys.executable, "-c", "print('next')"], 60, log_file=log_file)
    assert retcode == 0
    assert log_file.read_text(encoding="utf-8") == "previous\nnext\n"


def test_run_subprocess_log_keeps_output_tail(tmp_path, capsys):
    log_file = tmp_path / "nb.log"
    code = "import sys\nfor i in range(5000): print(f'line {i}')\nprint('last line')\nsys.exit(3)"
    retcode, _ = run_subprocess_with_timeout([sys.executable, "-c", code], 60, log_file=log_file)
    assert retcode == 3
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert lines == [f"line {i}" for i in range(5000)] + ["last line"]
    assert "last line" in capsys.readouterr().out


def test_run_subprocess_timeout(tmp_path, capsys):
    log_file = tmp_path / "nb.log"
    stats = {}
    code = "import time; print('started', flush=True); time.sleep(30)"
    retcode, duration = run_subprocess_with_timeout([sys.executable, "-c", code], 2, log_file=log_file, stats=stats)
    assert retcode == -42
    assert 2 <= duration < 20
    content = log_file.read_text(encoding="utf-8")
    assert content.startswith("started\n")
    assert content.endswith(TIMEOUT_MARKER + "\n")
    assert stats["peak_rss_bytes"] > 0
    assert "timeout reached (2s), killing process" in capsys.readouterr().out


def test_run_subprocess_without_dashboard_args(tmp_path, monkeypatch, capsys):
    def no_psutil(*args, **kwargs):
        raise AssertionError("psutil must not be used without stats")

    monkeypatch.setattr(validate_notebooks.psutil, "Process", no_psutil)
    monkeypatch.chdir(tmp_path)
    retcode, duration = run_subprocess_with_timeout([sys.executable, "-c", "print('plain')"], 60, description="Plain")
    assert retcode == 0 and duration > 0
    out = capsys.readouterr().out
    assert "Running Plain:" in out and "plain\n" in out
    assert list(tmp_path.iterdir()) == []

    retcode, _ = run_subprocess_with_timeout([sys.executable, "-c", "import sys; sys.exit(5)"], 60)
    assert retcode == 5


def test_run_subprocess_unwritable_log_does_not_fail(tmp_path, capsys):
    log_dir = tmp_path / "is_a_directory"
    log_dir.mkdir()
    retcode, _ = run_subprocess_with_timeout([sys.executable, "-c", "print('still runs')"], 60, log_file=log_dir, stats={})
    assert retcode == 0
    out = capsys.readouterr().out
    assert "still runs" in out and "WARNING: cannot open output log file" in out


def test_run_subprocess_harness_error(tmp_path, monkeypatch):
    def broken_popen(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(validate_notebooks.subprocess, "Popen", broken_popen)
    stats = {}
    retcode, _ = run_subprocess_with_timeout(["anything"], 60, log_file=tmp_path / "nb.log", stats=stats)
    assert retcode == -1
    assert stats["harness_error"] == "OSError: cannot spawn"

    retcode, _ = run_subprocess_with_timeout(["anything"], 60)
    assert retcode == -1


# --- prepare_test_plan ---------------------------------------------------------------------------------------------------


def make_notebooks(root: Path, names: list[str], patched: bool = True) -> Path:
    notebooks_dir = root / "notebooks"
    for name in names:
        notebook_dir = notebooks_dir / name
        notebook_dir.mkdir(parents=True, exist_ok=True)
        (notebook_dir / f"{name}.ipynb").write_text("{}", encoding="utf-8")
        if patched:
            (notebook_dir / f"test_{name}.ipynb").write_text("{}", encoding="utf-8")
    return notebooks_dir


@pytest.fixture
def skip_files(tmp_path):
    skip_config = tmp_path / "skipped.yml"
    skip_config.write_text(SKIP_CONFIG, encoding="utf-8")
    ignore_txt = tmp_path / "ignore.txt"
    ignore_txt.write_text("notebooks/ignored/ignored.ipynb\nnotebooks/skipcfg/skipcfg.ipynb\nnotebooks/outofscope/outofscope.ipynb\n", encoding="utf-8")
    return skip_config, ignore_txt


@pytest.mark.parametrize("moved", [False, True])
def test_prepare_test_plan_records_skip_reason(tmp_path, skip_files, moved):
    skip_config, ignore_txt = skip_files
    notebooks_dir = make_notebooks(tmp_path / "repo", NOTEBOOKS, patched=False)
    if moved:
        # Same layout as --move_notebooks_dir: an absolute notebooks directory outside of the repository
        notebooks_dir = Path(shutil.copytree(notebooks_dir, tmp_path / "moved" / "notebooks"))
    test_list = [f"notebooks/{name}/{name}.ipynb" for name in TESTING_LIST]
    test_plan = prepare_test_plan(
        {"os": "ubuntu-22.04", "python": "3.11", "device": "cpu"},
        test_list,
        str(skip_config),
        [str(ignore_txt)],
        notebooks_dir,
    )
    summary = {notebook.as_posix(): (report["status"], report["skip_reason"]) for notebook, report in test_plan.items()}
    assert summary == {
        "crash/crash.ipynb": ("", None),
        "fail/fail.ipynb": ("", None),
        "harness/harness.ipynb": ("", None),
        "ignored/ignored.ipynb": (NotebookStatus.SKIPPED, SkipReason.IGNORE_LIST),
        "missing/missing.ipynb": ("", None),
        "outofscope/outofscope.ipynb": (NotebookStatus.SKIPPED, None),
        "pass/pass.ipynb": ("", None),
        "skipcfg/skipcfg.ipynb": (NotebookStatus.SKIPPED, SkipReason.SKIP_CONFIG),
        "slow/slow.ipynb": ("", None),
    }
    assert all(report["path"] == notebooks_dir / notebook for notebook, report in test_plan.items())


def test_prepare_test_plan_without_test_list(tmp_path, skip_files):
    skip_config, _ = skip_files
    notebooks_dir = make_notebooks(tmp_path / "repo", ["a", "skipcfg"], patched=False)
    test_plan = prepare_test_plan({"os": "ubuntu-22.04", "python": "3.11", "device": "cpu"}, None, str(skip_config), ["notebooks/a/a.ipynb"], notebooks_dir)
    assert {n.as_posix(): (r["status"], r["skip_reason"]) for n, r in test_plan.items()} == {
        "a/a.ipynb": (NotebookStatus.SKIPPED, SkipReason.IGNORE_LIST),
        "skipcfg/skipcfg.ipynb": (NotebookStatus.SKIPPED, SkipReason.SKIP_CONFIG),
    }


# --- main() with a fake run_test -------------------------------------------------------------------------------------------


class FakeRunTest:
    """Simulates run_test() outcomes by notebook name."""

    def __init__(self):
        self.calls = []

    def __call__(self, notebook_path, root, timeout=7200, keep_artifacts=False, report_dir=".", source_venv_path=None, log_file=None, stats=None):
        name = Path(notebook_path).stem
        self.calls.append((name, log_file, stats))
        patched = f"test_{Path(notebook_path).name}"
        if name == "missing":
            return None
        if log_file is not None:
            Path(log_file).write_text(f"output of {name}\nTraceback (most recent call last):\nValueError: boom\n", encoding="utf-8")
        if stats is not None:
            stats["peak_rss_bytes"] = 1000
        if name == "crash":
            raise RuntimeError("harness exploded")
        (Path(report_dir) / f"test_{name}_env_after.txt").write_text(FREEZE, encoding="utf-8")
        if name == "harness":
            if stats is not None:
                stats["harness_error"] = "OSError: cannot spawn"
            return patched, -1, 0.5, "OpenVINO is missing", "OpenVINO is missing"
        return_codes = {"pass": 0, "fail": 1, "slow": -42}
        return patched, return_codes[name], 2.5, "2026.1.0", "2026.1.0"


@pytest.fixture
def main_setup(tmp_path, skip_files, monkeypatch):
    skip_config, ignore_txt = skip_files
    notebooks_dir = make_notebooks(tmp_path / "repo", NOTEBOOKS)
    fake = FakeRunTest()
    monkeypatch.setattr(validate_notebooks, "run_test", fake)

    def run_main(*extra_args, report_dir="report"):
        argv = [
            "validate_notebooks.py",
            "--notebooks_dir",
            str(notebooks_dir),
            "--ignore_config",
            str(skip_config),
            "--ignore_list",
            str(ignore_txt),
            "--test_list",
            *[f"notebooks/{name}/{name}.ipynb" for name in TESTING_LIST],
            "--os",
            "ubuntu-22.04",
            "--python",
            "3.11",
            "--device",
            "cpu",
            "--timeout",
            "1200",
            "--report_dir",
            str(tmp_path / report_dir),
            *extra_args,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        fake.calls.clear()
        return validate_notebooks.main()

    return run_main, fake, tmp_path


DASHBOARD_ARGS = ["--run_id", "123", "--run_attempt", "2", "--job_id", "", "--batch", "batch_1", "--runner_name", "", "--runner_label", "ubuntu-22.04-8-cores"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_main_writes_fragment(main_setup, capsys):
    run_main, fake, tmp_path = main_setup

    exit_code_without = run_main(report_dir="report_plain")
    assert all(log_file is None and stats is None for _, log_file, stats in fake.calls)
    capsys.readouterr()

    dashboard_dir = tmp_path / "dashboard"
    exit_code = run_main("--dashboard_dir", str(dashboard_dir), *DASHBOARD_ARGS)
    assert exit_code == exit_code_without == 1
    assert read_csv(tmp_path / "report" / "test_report.csv") == read_csv(tmp_path / "report_plain" / "test_report.csv")
    assert "WARNING: nightly dashboard" not in capsys.readouterr().out

    data = assert_valid_fragment(dashboard_dir / "fragment.json")
    job = data["job"]
    assert job["run_id"] == 123 and job["run_attempt"] == 2 and job["job_id"] is None
    assert job["batch"] == "batch_1" and job["runner_name"] is None and job["runner_label"] == "ubuntu-22.04-8-cores"
    assert (job["device"], job["os"], job["python"], job["timeout_s"], job["separate_venv"]) == ("cpu", "ubuntu-22.04", "3.11", 1200, False)
    assert job["complete"] is True and job["finished_at"]
    assert job["python_full_version"] and job["platform"]

    expected_planned = sorted(f"{name}/{name}.ipynb" for name in TESTING_LIST)
    assert data["planned"] == expected_planned

    results = {r["notebook"]: r for r in data["results"]}
    assert set(results) == set(expected_planned)
    assert {n: (r["status"], r["skip_reason"]) for n, r in results.items()} == {
        "pass/pass.ipynb": ("passed", None),
        "fail/fail.ipynb": ("failed", None),
        "slow/slow.ipynb": ("timeout", None),
        "missing/missing.ipynb": ("error", None),
        "crash/crash.ipynb": ("error", None),
        "harness/harness.ipynb": ("error", None),
        "skipcfg/skipcfg.ipynb": ("skipped", "skip_config"),
        "ignored/ignored.ipynb": ("skipped", "ignore_list"),
    }
    assert results["missing/missing.ipynb"]["error_hint"] == "Patched notebook not found or invalid notebook path"
    assert results["crash/crash.ipynb"]["error_hint"] == "RuntimeError: harness exploded"
    assert results["harness/harness.ipynb"]["error_hint"] == "OSError: cannot spawn"
    assert results["harness/harness.ipynb"]["return_code"] == -1
    assert results["harness/harness.ipynb"]["openvino_before"] is None

    passed = results["pass/pass.ipynb"]
    assert passed["return_code"] == 0 and passed["duration_s"] == 2.5 and passed["peak_rss_bytes"] == 1000
    assert passed["openvino_before"] == passed["openvino_after"] == "2026.1.0"
    assert passed["log_file"] is None and not (dashboard_dir / "logs" / "pass__pass.log").exists()
    env_id = env_id_for(parse_pip_freeze(FREEZE))
    assert passed["env_id"] == env_id and passed["packages"] == {"openvino": "2026.1.0", "numpy": "2.2.6"}
    assert (dashboard_dir / "envs" / f"{env_id}.txt").is_file()
    assert passed["started_at"] <= passed["finished_at"]

    for name in ("fail", "slow", "crash", "harness"):
        result = results[f"{name}/{name}.ipynb"]
        assert result["log_file"] == f"logs/{name}__{name}.log"
        assert "ValueError: boom" in (dashboard_dir / result["log_file"]).read_text(encoding="utf-8")
    assert results["fail/fail.ipynb"]["return_code"] == 1
    assert results["slow/slow.ipynb"]["return_code"] == -42
    assert results["missing/missing.ipynb"]["log_file"] is None and results["missing/missing.ipynb"]["return_code"] is None
    assert results["crash/crash.ipynb"]["return_code"] is None and results["crash/crash.ipynb"]["peak_rss_bytes"] == 1000
    assert results["skipcfg/skipcfg.ipynb"]["started_at"] is None


def test_main_early_stop_finishes_fragment(main_setup):
    run_main, fake, tmp_path = main_setup
    dashboard_dir = tmp_path / "dashboard"
    run_main("--dashboard_dir", str(dashboard_dir), "--early_stop")
    data = assert_valid_fragment(dashboard_dir / "fragment.json")
    assert data["job"]["complete"] is True
    assert len(fake.calls) == 1
    executed = [r for r in data["results"] if r["status"] != "skipped"]
    assert [r["notebook"] for r in executed] == ["crash/crash.ipynb"]
    assert len(data["results"]) == 3


def test_main_fragment_failures_do_not_affect_testing(main_setup, monkeypatch, capsys):
    run_main, fake, tmp_path = main_setup
    exit_code_without = run_main(report_dir="report_plain")
    expected_csv = read_csv(tmp_path / "report_plain" / "test_report.csv")

    class BrokenWriter:
        def __init__(self, *args, **kwargs):
            raise PermissionError("read-only file system")

    monkeypatch.setattr(fragment_module, "FragmentWriter", BrokenWriter)
    capsys.readouterr()
    assert run_main("--dashboard_dir", str(tmp_path / "dashboard_broken"), report_dir="report_broken") == exit_code_without
    assert "WARNING: nightly dashboard fragment operation 'create_fragment_writer' failed: PermissionError" in capsys.readouterr().out
    assert all(log_file is None for _, log_file, _ in fake.calls)
    assert read_csv(tmp_path / "report_broken" / "test_report.csv") == expected_csv

    monkeypatch.undo()
    monkeypatch.setattr(validate_notebooks, "run_test", fake)

    original_add_result = fragment_module.FragmentWriter.add_result

    def broken_add_result(self, notebook, status, **kwargs):
        if status != "skipped":
            raise OSError("disk full")
        return original_add_result(self, notebook, status, **kwargs)

    def broken_finish(self):
        raise OSError("disk full")

    monkeypatch.setattr(fragment_module.FragmentWriter, "add_result", broken_add_result)
    monkeypatch.setattr(fragment_module.FragmentWriter, "finish", broken_finish)
    assert run_main("--dashboard_dir", str(tmp_path / "dashboard_full"), report_dir="report_full") == exit_code_without
    out = capsys.readouterr().out
    assert "WARNING: nightly dashboard fragment operation 'record_fragment_result' failed: OSError: disk full" in out
    assert "WARNING: nightly dashboard fragment operation 'broken_finish' failed: OSError: disk full" in out
    assert assert_valid_fragment(tmp_path / "dashboard_full" / "fragment.json")["job"]["complete"] is False
    assert read_csv(tmp_path / "report_full" / "test_report.csv") == expected_csv


# --- standalone copy without nightly_report (as in the Docker image) ------------------------------------------------------

# Mirrors the Dockerfile: only these scripts are copied to /tmp/scripts
DOCKER_SCRIPTS = ("validate_notebooks.py", "validation_config.py", "skip_resolution.py")

STANDALONE_MAIN = """
import sys

import validate_notebooks

sys.argv = ["validate_notebooks.py", *sys.argv[1:]]
exit_code = validate_notebooks.main()
print("imported nightly_report:", any(name.split(".")[0] == "nightly_report" for name in sys.modules))
print("exit code:", exit_code)
"""


@pytest.fixture
def standalone_scripts(tmp_path) -> Path:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for name in DOCKER_SCRIPTS:
        shutil.copy(CI_DIR / name, scripts_dir / name)
    return scripts_dir


def run_standalone(scripts_dir: Path, *args) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    return subprocess.run(  # nosec B603 - fixed command built from test data
        [sys.executable, *map(str, args)], cwd=scripts_dir, env=env, capture_output=True, text=True, timeout=120
    )


def test_standalone_script_help(standalone_scripts):
    result = run_standalone(standalone_scripts, standalone_scripts / "validate_notebooks.py", "--help")
    assert result.returncode == 0, result.stderr
    assert "--dashboard_dir" in result.stdout


@pytest.mark.parametrize("dashboard", [False, True])
def test_standalone_script_main(tmp_path, standalone_scripts, dashboard):
    notebooks_dir = make_notebooks(tmp_path / "repo", ["a", "b"])
    args = [
        "--notebooks_dir",
        notebooks_dir,
        "--ignore_config",
        tmp_path / "no_skips.yml",
        "--ignore_list",
        "notebooks/a/a.ipynb",
        "notebooks/b/b.ipynb",
        "--os",
        "ubuntu-22.04",
        "--python",
        "3.11",
        "--device",
        "cpu",
        "--report_dir",
        tmp_path / "report",
    ]
    if dashboard:
        args += ["--dashboard_dir", tmp_path / "dashboard"]
    result = run_standalone(standalone_scripts, "-c", STANDALONE_MAIN, *args)
    assert result.returncode == 0, result.stderr
    assert "exit code: 0" in result.stdout
    assert "imported nightly_report: False" in result.stdout
    assert (tmp_path / "report" / "test_report.csv").is_file()
    if dashboard:
        assert "WARNING: nightly dashboard fragment operation 'create_fragment_writer' failed: ModuleNotFoundError" in result.stdout
        assert not (tmp_path / "dashboard").exists()
    else:
        assert "WARNING" not in result.stdout


# --- end-to-end with treon -------------------------------------------------------------------------------------------------


def write_notebook(path: Path, source: str):
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell("print('cell output')"), nbformat.v4.new_code_cell(source)]
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, str(path))


def test_main_end_to_end_with_treon(tmp_path, monkeypatch, restore_environ):
    pytest.importorskip("treon")
    pytest.importorskip("ipykernel")
    notebooks_dir = tmp_path / "repo" / "notebooks"
    for name, source in (("ok", "x = 1 + 1\nprint(x)"), ("bad", "raise ValueError('boom from notebook')")):
        write_notebook(notebooks_dir / name / f"{name}.ipynb", source)
        shutil.copy(notebooks_dir / name / f"{name}.ipynb", notebooks_dir / name / f"test_{name}.ipynb")

    monkeypatch.setattr(validate_notebooks, "print_disk_usage", lambda *args, **kwargs: None)
    dashboard_dir = tmp_path / "dashboard"
    argv = [
        "validate_notebooks.py",
        "--notebooks_dir",
        str(notebooks_dir),
        "--ignore_config",
        str(tmp_path / "no_skips.yml"),
        "--os",
        "ubuntu-22.04",
        "--python",
        "3.11",
        "--device",
        "cpu",
        "--timeout",
        "300",
        "--report_dir",
        str(tmp_path / "report"),
        "--dashboard_dir",
        str(dashboard_dir),
        "--run_id",
        "1",
        "--job_id",
        "",
        "--batch",
        "batch_0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert validate_notebooks.main() == 1

    rows = {row["name"]: row["status"] for row in read_csv(tmp_path / "report" / "test_report.csv")}
    assert rows == {"bad/bad.ipynb": NotebookStatus.FAILED, "ok/ok.ipynb": NotebookStatus.SUCCESS}

    data = assert_valid_fragment(dashboard_dir / "fragment.json")
    assert data["job"]["complete"] is True
    assert data["planned"] == ["bad/bad.ipynb", "ok/ok.ipynb"]
    results = {r["notebook"]: r for r in data["results"]}
    assert results["ok/ok.ipynb"]["status"] == "passed" and results["ok/ok.ipynb"]["log_file"] is None
    assert results["bad/bad.ipynb"]["status"] == "failed" and results["bad/bad.ipynb"]["return_code"] != 0
    log = (dashboard_dir / results["bad/bad.ipynb"]["log_file"]).read_text(encoding="utf-8")
    assert "boom from notebook" in log
    assert sorted(p.name for p in (dashboard_dir / "logs").iterdir()) == ["bad__bad.log"]
    for result in results.values():
        assert result["peak_rss_bytes"] and result["peak_rss_bytes"] > 0
        assert result["env_id"] and (dashboard_dir / "envs" / f"{result['env_id']}.txt").is_file()
        assert result["openvino_before"] is None and result["duration_s"] > 0
