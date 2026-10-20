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


@pytest.mark.parametrize("count", [1, 2, 5, 7])
def test_precommit_nonempty_matrix(count, tmp_path, notebooks_dir, monkeypatch, capsys):
    selected = expected_paths(notebooks_dir)[:count]
    output = tmp_path / "outputs"
    monkeypatch.setenv("CHANGED_FILES", json.dumps(selected + selected))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    run_split(monkeypatch, capsys, "--precommit", "--notebooks_dir", notebooks_dir, "--num_batches", 5, "--seed", 42)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    matrix = json.loads(values["matrix"])
    assert values["has_tests"] == "true"
    assert len(matrix) == min(count, 5)
    assert all(batch["test_list"] for batch in matrix)
    assert sorted(path for batch in matrix for path in batch["test_list"].splitlines()) == selected


@pytest.mark.parametrize(
    "changed, expected",
    [
        (["requirements.txt"], NOTEBOOKS),
        (["notebooks/b/helper.py"], ["b/b.ipynb", "b/b-second.ipynb"]),
        (["notebooks/b/nested/helper.py"], ["b/b.ipynb", "b/b-second.ipynb"]),
        (["README.md"], ["a/a.ipynb"]),
        (["notebooks/deleted/deleted.ipynb"], ["a/a.ipynb"]),
        (["notebooks/b/old-name.ipynb", "notebooks/b/b-second.ipynb"], ["b/b-second.ipynb"]),
        (["notebooks/a/test_a.ipynb", "notebooks/b/.venv/helper.py", "notebooks/b/.ipynb_checkpoints/helper.py"], ["a/a.ipynb"]),
    ],
)
def test_precommit_selection(changed, expected, tmp_path, notebooks_dir, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fallback = tmp_path / "fallback.txt"
    fallback.write_text("notebooks/a/a.ipynb\n", encoding="utf-8")
    result = split_notebooks.select_precommit_notebooks(notebooks_dir, changed, fallback)
    assert result == sorted(str(notebooks_dir / path) for path in expected)


@pytest.mark.parametrize("content", ["", "notebooks/missing.ipynb\n"])
def test_precommit_invalid_fallback(content, tmp_path, notebooks_dir, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fallback = tmp_path / "fallback.txt"
    fallback.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Fallback list"):
        split_notebooks.select_precommit_notebooks(notebooks_dir, [], fallback)


def test_precommit_path_with_spaces(tmp_path, notebooks_dir, monkeypatch, capsys):
    path = notebooks_dir / "with spaces" / "a notebook.ipynb"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CHANGED_FILES", json.dumps([str(path)]))
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    run_split(monkeypatch, capsys, "--precommit", "--notebooks_dir", notebooks_dir, "--num_batches", 5)
    matrix = json.loads(output.read_text().splitlines()[0].split("=", 1)[1])
    assert matrix == [{"name": "batch_0", "test_list": str(path)}]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_batch_count(value, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run_split(monkeypatch, capsys, "--num_batches", value)


@pytest.mark.parametrize("value", ['{"path": "notebooks/a.ipynb"}', "[1]", "not json"])
def test_precommit_invalid_changed_files(value, notebooks_dir, monkeypatch, capsys):
    monkeypatch.setenv("CHANGED_FILES", value)
    with pytest.raises(ValueError):
        run_split(monkeypatch, capsys, "--precommit", "--notebooks_dir", notebooks_dir)


def test_precommit_missing_fallback(tmp_path, notebooks_dir):
    with pytest.raises(FileNotFoundError):
        split_notebooks.select_precommit_notebooks(notebooks_dir, [], tmp_path / "missing.txt")


def test_precommit_no_notebooks(tmp_path):
    root = tmp_path / "notebooks"
    root.mkdir()
    with pytest.raises(ValueError, match="No notebooks"):
        split_notebooks.select_precommit_notebooks(root, [str(tmp_path / "requirements.txt")], tmp_path / "fallback.txt")


def test_collect_excludes_generated_notebooks(notebooks_dir):
    for directory in (".venv", ".ipynb_checkpoints"):
        path = notebooks_dir / "b" / directory / "generated.ipynb"
        path.parent.mkdir()
        path.write_text("{}", encoding="utf-8")
    assert split_notebooks.collect_notebooks(notebooks_dir) == expected_paths(notebooks_dir)
