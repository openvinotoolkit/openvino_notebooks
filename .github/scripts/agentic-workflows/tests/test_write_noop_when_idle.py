import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import write_noop_when_idle as noop


def test_no_work_emits_a_noop_with_the_collector_reason(tmp_path):
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"has_work": False, "reason": 'all "failures" are tracked\nalready'}), encoding="utf-8")
    output_path = tmp_path / "missing" / "safeoutputs" / "outputs.jsonl"

    assert noop.write_noop_when_idle(status_path, output_path) is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "type": "noop",
        "message": 'Nightly remediation skipped: all "failures" are tracked\nalready',
    }


def test_work_does_not_create_a_noop(tmp_path):
    status_path = tmp_path / "status.json"
    status_path.write_text('{"has_work": true, "reason": "one group"}', encoding="utf-8")
    output_path = tmp_path / "missing" / "outputs.jsonl"

    assert noop.write_noop_when_idle(status_path, output_path) is False
    assert not output_path.exists()


@pytest.mark.parametrize(
    "content",
    ["", "{", "{}", '{"has_work": "false"}', '{"has_work": false}', '{"has_work": false, "reason": ""}'],
)
def test_invalid_status_fails_instead_of_emitting_a_noop(tmp_path, content):
    status_path = tmp_path / "status.json"
    status_path.write_text(content, encoding="utf-8")
    output_path = tmp_path / "outputs.jsonl"

    with pytest.raises(ValueError):
        noop.write_noop_when_idle(status_path, output_path)
    assert not output_path.exists()


def test_missing_status_fails(tmp_path):
    with pytest.raises(ValueError, match="missing or empty"):
        noop.write_noop_when_idle(tmp_path / "missing.json", tmp_path / "outputs.jsonl")


def test_cli_creates_parent_and_appends_to_the_bound_safe_output(tmp_path):
    digest_dir = tmp_path / "digest"
    digest_dir.mkdir()
    (digest_dir / "status.json").write_text('{"has_work": false, "reason": "nothing selected"}', encoding="utf-8")
    output_path = tmp_path / "runner" / "safeoutputs" / "outputs.jsonl"
    env = {**os.environ, "DIGEST_DIR": str(digest_dir), "GH_AW_SAFE_OUTPUTS": str(output_path)}
    script = Path(noop.__file__)

    subprocess.run([sys.executable, str(script)], env=env, check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(script)], env=env, check=True, capture_output=True, text=True)

    entries = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert entries == [{"type": "noop", "message": "Nightly remediation skipped: nothing selected"}] * 2
