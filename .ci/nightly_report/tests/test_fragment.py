import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from nightly_report import fragment as fragment_module
from nightly_report.constants import KEY_PACKAGES, SCHEMA_VERSION, SkipReason, Status
from nightly_report.fragment import FragmentWriter, cap_log_file, env_id_for, key_packages, parse_pip_freeze, utc_now_iso

CI_DIR = Path(__file__).resolve().parents[2]
FRAGMENT_SCHEMA = json.loads((CI_DIR / "nightly_report" / "schema" / "fragment.schema.json").read_text(encoding="utf-8"))

FREEZE = """\
# comment line
openvino==2026.1.0
openvino-tokenizers==2026.1.0.0
Torch==2.8.0+cpu
huggingface_hub==0.35.0
numpy==2.2.6
nncf @ git+https://github.com/openvinotoolkit/nncf.git@abc123
-e git+https://github.com/example/editable-pkg.git@deadbeef#egg=Editable_Pkg&subdirectory=src
-e /local/path/without/egg

zope.interface===7.0
"""

JOB = {
    "run_id": 123,
    "run_attempt": 1,
    "job_id": None,
    "batch": "batch_0",
    "device": "cpu",
    "os": "ubuntu-22.04",
    "python": "3.11",
    "python_full_version": "3.11.9",
    "platform": "Linux-x86_64",
    "runner_name": "runner-1",
    "runner_label": "ubuntu-22.04-16-cores",
    "container_image": None,
    "timeout_s": 1200,
    "separate_venv": True,
    "openvino_base_version": "Openvino is missing",
}


def assert_valid_fragment(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator(FRAGMENT_SCHEMA, format_checker=Draft202012Validator.FORMAT_CHECKER).validate(data)
    return data


def write_log(writer: FragmentWriter, notebook: str, content: bytes) -> Path:
    path = writer.log_path(notebook)
    path.write_bytes(content)
    return path


def test_utc_now_iso_format():
    value = utc_now_iso()
    assert value.endswith("Z") and "+" not in value and len(value) == len("2026-01-01T00:00:00Z")


def test_parse_pip_freeze():
    freeze = parse_pip_freeze(FREEZE)
    assert freeze == {
        "openvino": "2026.1.0",
        "openvino-tokenizers": "2026.1.0.0",
        "torch": "2.8.0+cpu",
        "huggingface-hub": "0.35.0",
        "numpy": "2.2.6",
        "nncf": "@ git+https://github.com/openvinotoolkit/nncf.git@abc123",
        "editable-pkg": "-e git+https://github.com/example/editable-pkg.git@deadbeef#egg=Editable_Pkg&subdirectory=src",
        "zope-interface": "===7.0",
    }


def test_parse_pip_freeze_ignores_blanks_and_options():
    assert parse_pip_freeze("\n  \n# only comments\n--index-url https://example.com\n") == {}
    assert parse_pip_freeze("Foo.Bar_baz>=1.0\n") == {"foo-bar-baz": ">=1.0"}


def test_key_packages_and_env_id():
    freeze = parse_pip_freeze(FREEZE)
    packages = key_packages(freeze)
    assert set(packages) <= set(KEY_PACKAGES)
    assert packages == {
        "openvino": "2026.1.0",
        "openvino-tokenizers": "2026.1.0.0",
        "nncf": "@ git+https://github.com/openvinotoolkit/nncf.git@abc123",
        "torch": "2.8.0+cpu",
        "huggingface-hub": "0.35.0",
        "numpy": "2.2.6",
    }
    env_id = env_id_for(freeze)
    assert len(env_id) == 12 and int(env_id, 16) >= 0
    assert env_id == env_id_for(dict(reversed(list(freeze.items()))))
    assert env_id != env_id_for({**freeze, "numpy": "2.2.5"})


def test_cap_log_file(tmp_path):
    small = tmp_path / "small.log"
    small.write_bytes(b"x" * 100)
    cap_log_file(small, max_bytes=100, head_bytes=10)
    assert small.read_bytes() == b"x" * 100

    big = tmp_path / "big.log"
    content = bytes(range(256)) * 40  # 10240 bytes, not valid utf-8
    big.write_bytes(content)
    cap_log_file(big, max_bytes=1000, head_bytes=200)
    capped = big.read_bytes()
    marker = b"\n...[truncated 9240 bytes]...\n"
    assert capped == content[:200] + marker + content[-800:]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["big.log", "small.log"]


def test_fragment_writer_end_to_end(tmp_path):
    out_dir = tmp_path / "dashboard"
    planned = ["a/a.ipynb", "b/b.ipynb", "c/c.ipynb", "d/d.ipynb", "e/e.ipynb"]
    writer = FragmentWriter(out_dir, {**JOB, "complete": True, "started_at": "ignored"}, planned)
    fragment_path = out_dir / "fragment.json"

    data = assert_valid_fragment(fragment_path)
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["planned"] == planned
    assert data["results"] == []
    assert data["job"]["complete"] is False and data["job"]["finished_at"] is None
    assert data["job"]["started_at"] != "ignored"
    assert data["job"]["openvino_base_version"] is None
    assert (out_dir / "logs").is_dir() and (out_dir / "envs").is_dir()

    freeze_file = tmp_path / "test_a_env_after.txt"
    freeze_file.write_text(FREEZE, encoding="utf-8")
    env_id = env_id_for(parse_pip_freeze(FREEZE))

    writer.add_result("e/e.ipynb", Status.SKIPPED, skip_reason=SkipReason.SKIP_CONFIG)

    passed_log = write_log(writer, "a/a.ipynb", b"all good\n")
    passed = writer.add_result(
        "a/a.ipynb",
        Status.PASSED,
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        duration_s=12.5,
        return_code=0,
        peak_rss_bytes=1024,
        openvino_before="2026.1.0",
        openvino_after="2026.1.0",
        env_freeze_file=freeze_file,
    )
    assert not passed_log.exists()
    assert passed["log_file"] is None
    assert passed["env_id"] == env_id
    assert passed["packages"]["openvino"] == "2026.1.0"
    assert (out_dir / "envs" / f"{env_id}.txt").read_text(encoding="utf-8") == FREEZE

    failed_log = write_log(writer, "b/b.ipynb", b"y" * (fragment_module.LOG_MAX_BYTES + 1000))
    failed = writer.add_result(
        "b/b.ipynb",
        Status.FAILED,
        duration_s=3,
        return_code=1,
        openvino_before="OpenVINO is missing",
        openvino_after="N/A",
        env_freeze_file=freeze_file,
    )
    assert failed_log.exists()
    assert failed["log_file"] == "logs/b__b.log"
    assert failed_log.stat().st_size < fragment_module.LOG_MAX_BYTES + 100
    assert b"...[truncated 1000 bytes]..." in failed_log.read_bytes()
    assert failed["openvino_before"] is None and failed["openvino_after"] is None
    assert failed["env_id"] == env_id
    assert sorted(p.name for p in (out_dir / "envs").iterdir()) == [f"{env_id}.txt"]

    error = writer.add_result("c/c.ipynb", Status.ERROR, env_freeze_file=tmp_path / "missing.txt", error_hint="boom")
    assert error["packages"] == {} and error["env_id"] is None and error["log_file"] is None

    data = assert_valid_fragment(fragment_path)
    assert [r["notebook"] for r in data["results"]] == ["e/e.ipynb", "a/a.ipynb", "b/b.ipynb", "c/c.ipynb"]
    assert data["job"]["complete"] is False

    writer.finish()
    data = assert_valid_fragment(fragment_path)
    assert data["job"]["complete"] is True and data["job"]["finished_at"] is not None
    assert {r["notebook"]: r["status"] for r in data["results"]} == {
        "e/e.ipynb": "skipped",
        "a/a.ipynb": "passed",
        "b/b.ipynb": "failed",
        "c/c.ipynb": "error",
    }
    assert data["results"][0]["skip_reason"] == "skip_config"
    assert not [p for p in out_dir.iterdir() if p.name.endswith(".tmp")]


def test_add_result_replaces_existing_result(tmp_path):
    writer = FragmentWriter(tmp_path, JOB, ["a/a.ipynb", "b/b.ipynb"])
    writer.add_result("a/a.ipynb", Status.FAILED, return_code=1)
    writer.add_result("b/b.ipynb", Status.PASSED, return_code=0)
    writer.add_result("a/a.ipynb", Status.PASSED, return_code=0)
    data = assert_valid_fragment(tmp_path / "fragment.json")
    assert [(r["notebook"], r["status"]) for r in data["results"]] == [("a/a.ipynb", "passed"), ("b/b.ipynb", "passed")]


def test_windows_style_notebook_paths_are_posix(tmp_path):
    writer = FragmentWriter(tmp_path, JOB, ["a\\a.ipynb"])
    log = write_log(writer, "a\\a.ipynb", b"error\n")
    assert log.name == "a__a.log"
    result = writer.add_result("a\\a.ipynb", Status.FAILED, return_code=1)
    assert result["notebook"] == "a/a.ipynb" and result["log_file"] == "logs/a__a.log"
    assert assert_valid_fragment(tmp_path / "fragment.json")["planned"] == ["a/a.ipynb"]


def test_save_is_atomic(tmp_path, monkeypatch):
    writer = FragmentWriter(tmp_path, JOB, ["a/a.ipynb"])
    before = (tmp_path / "fragment.json").read_bytes()

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(fragment_module.os, "replace", failing_replace)
    with pytest.raises(OSError):
        writer.add_result("a/a.ipynb", Status.PASSED)
    assert (tmp_path / "fragment.json").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["envs", "fragment.json", "logs"]
