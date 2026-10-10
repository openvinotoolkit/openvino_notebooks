import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import split_notebooks
from nightly_report.constants import SCHEMA_VERSION

CI_DIR = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = json.loads((CI_DIR / "nightly_report" / "schema" / "plan.schema.json").read_text(encoding="utf-8"))

NOTEBOOKS = [
    "a/a.ipynb",
    "b/b.ipynb",
    "b/b-second.ipynb",
    "c/c.ipynb",
    "d/nested/d.ipynb",
    "e/e.ipynb",
    "f/f.ipynb",
]


@pytest.fixture
def notebooks_dir(tmp_path) -> Path:
    root = tmp_path / "notebooks"
    for notebook in NOTEBOOKS:
        path = root / notebook
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    # Patched notebooks are not part of the plan
    (root / "a" / "test_a.ipynb").write_text("{}", encoding="utf-8")
    return root


def run_split(monkeypatch, capsys, *argv) -> str:
    monkeypatch.setattr(sys, "argv", ["split_notebooks.py", *map(str, argv)])
    split_notebooks.main()
    return capsys.readouterr().out


def expected_paths(notebooks_dir: Path) -> list[str]:
    return sorted((notebooks_dir / notebook).as_posix() for notebook in NOTEBOOKS)


def test_plan_file(tmp_path, notebooks_dir, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    plan_file = tmp_path / "out" / "plan.json"
    run_split(monkeypatch, capsys, "--notebooks_dir", notebooks_dir, "--num_batches", 3, "--seed", 42, "--plan_file", plan_file)

    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    Draft202012Validator(PLAN_SCHEMA).validate(plan)
    assert plan["schema_version"] == SCHEMA_VERSION
    assert plan["seed"] == "42"
    assert plan["num_batches"] == 3
    assert plan["notebooks_dir"] == notebooks_dir.as_posix()
    assert plan["notebooks_total"] == len(NOTEBOOKS)
    assert sorted(plan["batches"]) == ["batch_0", "batch_1", "batch_2"]
    all_planned = [notebook for batch in plan["batches"].values() for notebook in batch]
    assert sorted(all_planned) == expected_paths(notebooks_dir)
    assert len(all_planned) == len(set(all_planned))


def test_plan_matches_github_output(tmp_path, notebooks_dir, monkeypatch, capsys):
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    plan_file = tmp_path / "plan.json"
    run_split(monkeypatch, capsys, "--notebooks_dir", notebooks_dir, "--num_batches", 2, "--plan_file", plan_file)

    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    Draft202012Validator(PLAN_SCHEMA).validate(plan)
    assert plan["seed"] is None

    outputs = {}
    lines = github_output.read_text().splitlines()
    while lines:
        name = lines.pop(0).split("<<")[0]
        values = []
        while (line := lines.pop(0)) != "BATCH_EOF":
            values.append(line)
        outputs[name] = [Path(v).as_posix() for v in values]
    assert outputs == plan["batches"]


def test_stdout_unchanged_by_plan_file(tmp_path, notebooks_dir, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    without_plan = run_split(monkeypatch, capsys, "--notebooks_dir", notebooks_dir, "--num_batches", 3, "--seed", 7)
    assert not list(tmp_path.glob("*.json"))
    plan_file = tmp_path / "plan.json"
    with_plan = run_split(monkeypatch, capsys, "--notebooks_dir", notebooks_dir, "--num_batches", 3, "--seed", 7, "--plan_file", plan_file)
    assert with_plan.startswith(without_plan)
    assert with_plan[len(without_plan) :].strip() == f"Batch plan written to {plan_file}"


def test_more_batches_than_notebooks(tmp_path, notebooks_dir, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    plan_file = tmp_path / "plan.json"
    run_split(monkeypatch, capsys, "--notebooks_dir", notebooks_dir, "--num_batches", 10, "--seed", 1, "--plan_file", plan_file)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    Draft202012Validator(PLAN_SCHEMA).validate(plan)
    assert len(plan["batches"]) == 10
    assert sum(len(batch) for batch in plan["batches"].values()) == len(NOTEBOOKS)


def test_plan_file_failure_does_not_fail_split(tmp_path, notebooks_dir, monkeypatch, capsys):
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("", encoding="utf-8")
    plan_file = blocker / "plan.json"
    out = run_split(monkeypatch, capsys, "--notebooks_dir", notebooks_dir, "--num_batches", 2, "--seed", 3, "--plan_file", plan_file)

    assert f"WARNING: failed to write batch plan to {plan_file}" in out
    assert not plan_file.exists()
    content = github_output.read_text()
    assert content.count("<<BATCH_EOF") == 2
    written = [line for line in content.splitlines() if line.endswith(".ipynb")]
    assert sorted(Path(line).as_posix() for line in written) == expected_paths(notebooks_dir)


def write_list(tmp_path: Path, *lines: str) -> Path:
    list_file = tmp_path / "sample.txt"
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_file


def test_notebooks_list(tmp_path, notebooks_dir, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    list_file = write_list(
        tmp_path, "# sample", "notebooks/a/a.ipynb", "", "notebooks/b/b-second.ipynb  # comment", "notebooks/d/nested/d.ipynb", "notebooks/a/a.ipynb"
    )
    plan_file = tmp_path / "plan.json"
    out = run_split(monkeypatch, capsys, "--num_batches", 2, "--seed", 5, "--plan_file", plan_file, "--notebooks_list", list_file)

    assert f"Using 3 notebooks from {list_file}" in out
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    Draft202012Validator(PLAN_SCHEMA).validate(plan)
    assert plan["notebooks_total"] == 3
    planned = sorted(notebook for batch in plan["batches"].values() for notebook in batch)
    assert planned == ["notebooks/a/a.ipynb", "notebooks/b/b-second.ipynb", "notebooks/d/nested/d.ipynb"]
    # Same format as the discovered notebooks, so the split is reproducible for a seed.
    assert split_notebooks.read_notebooks_list(list_file, Path("notebooks")) == [str(Path(notebook)) for notebook in planned]


@pytest.mark.parametrize(
    "entry",
    ["notebooks/missing/missing.ipynb", "notebooks/a/test_a.ipynb", "notebooks/a", "other/x.ipynb"],
)
def test_notebooks_list_rejects_invalid_entries(tmp_path, notebooks_dir, monkeypatch, capsys, entry):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "x.ipynb").write_text("{}", encoding="utf-8")
    list_file = write_list(tmp_path, "notebooks/a/a.ipynb", "notebooks/b/b.ipynb", entry)
    with pytest.raises(SystemExit) as error:
        run_split(monkeypatch, capsys, "--num_batches", 2, "--notebooks_list", list_file)
    assert error.value.code == 2
    assert entry in capsys.readouterr().err


def test_notebooks_list_smaller_than_batches_fails(tmp_path, notebooks_dir, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    list_file = write_list(tmp_path, "notebooks/a/a.ipynb", "notebooks/b/b.ipynb")
    with pytest.raises(SystemExit) as error:
        run_split(monkeypatch, capsys, "--num_batches", 3, "--notebooks_list", list_file)
    assert error.value.code == 2
    assert "fewer than --num_batches 3" in capsys.readouterr().err
    assert not github_output.exists()


def test_nightly_sample_notebooks_list_is_valid(monkeypatch):
    repo_root = CI_DIR.parent
    monkeypatch.chdir(repo_root)
    notebooks = split_notebooks.read_notebooks_list(CI_DIR / "nightly_sample_notebooks.txt", Path("notebooks"))
    assert 3 <= len(notebooks) <= 20
