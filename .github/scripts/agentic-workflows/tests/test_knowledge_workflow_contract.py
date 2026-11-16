import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import knowledge_base as kb
from nightly_builders import shell_command_allowed
from test_knowledge_base import boundary_documents, invalid_documents
import update_knowledge_base as ukb

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = (REPO_ROOT / ".github/workflows/nightly-issue-remediator.md").read_text(encoding="utf-8")
LOCK = (REPO_ROOT / ".github/workflows/nightly-issue-remediator.lock.yml").read_text(encoding="utf-8")


def validation_script() -> str:
    """The `validation.script` block of the workflow's repo-memory configuration."""
    lines = WORKFLOW.split("\n---\n", 1)[0].splitlines()
    start = lines.index("      script: |") + 1
    body = []
    for line in lines[start:]:
        if line.strip() and not line.startswith(" " * 8):
            break
        body.append(line[8:])
    return "\n".join(body).strip() + "\n"


def run_validation_script(tmp_path, document):
    memory = tmp_path / "memory"
    shutil.rmtree(memory, ignore_errors=True)
    memory.mkdir()
    if document is not None:
        (memory / kb.KNOWLEDGE_FILE).write_text(document, encoding="utf-8")
    # Mirrors the wrapper of gh-aw's repo-memory validation runner: the script is the body of an async function.
    wrapper = (
        '"use strict";\nconst fs = require("fs");\nconst path = require("path");\n'
        f"const memoryRoot = {json.dumps(str(memory))};\n"
        "(async () => {\n  return await (async () => {\n" + validation_script() + "\n  })();\n})()\n"
        "  .then((result) => { if (result === false) process.exit(1); })\n"
        "  .catch((error) => { console.error(error.message); process.exit(1); });\n"
    )
    script = tmp_path / "validator.cjs"
    script.write_text(wrapper, encoding="utf-8")
    return subprocess.run(["node", str(script)], cwd=memory, capture_output=True, text=True, timeout=60)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@needs_node
def test_repo_memory_validation_script_accepts_what_the_updater_writes(tmp_path):
    assert run_validation_script(tmp_path, None).returncode == 0
    for name, document in boundary_documents().items():
        result = run_validation_script(tmp_path, document)
        assert result.returncode == 0, f"{name}: {result.stderr}"


@needs_node
def test_repo_memory_validation_script_rejects_what_the_python_reader_rejects(tmp_path):
    for name, document in invalid_documents().items():
        result = run_validation_script(tmp_path, document)
        assert result.returncode != 0, f"accepted: {name}"
        assert result.stderr.strip(), name


def test_compiled_workflow_embeds_the_validation_script_from_the_source():
    encoded = re.search(r"VALIDATION_SCRIPT_B64: (\S+)", LOCK).group(1)
    assert base64.b64decode(encoded).decode("utf-8").strip() == validation_script().strip()
    assert "VALIDATION_SCRIPT_REQUIRED: 'true'" in LOCK


def test_repo_memory_configuration_matches_the_knowledge_base_module():
    assert f"branch-name: {kb.KNOWLEDGE_BRANCH}" in WORKFLOW
    assert f'file-glob: ["{kb.KNOWLEDGE_FILE}"]' in WORKFLOW
    assert f"BRANCH_NAME: {kb.KNOWLEDGE_BRANCH}" in LOCK
    assert f'FILE_GLOB_FILTER: "{kb.KNOWLEDGE_FILE}"' in LOCK
    assert "MEMORY_DIR: /tmp/gh-aw/repo-memory/default" in LOCK
    assert "ALLOWED_EXTENSIONS: '[\".json\"]'" in LOCK
    # The cap on one file must leave room for the size the updater keeps the file under.
    assert f"max-file-size: {1024 * 1024}" in WORKFLOW and kb.MAX_FILE_BYTES < 1024 * 1024
    for limit in (f"> {kb.MAX_NOTEBOOKS}", f"> {kb.MAX_SIGNATURES}", f"> {kb.MAX_ATTEMPTS}", f"> {kb.TEXT_LIMIT}"):
        assert limit in validation_script(), limit


def test_job_level_env_does_not_use_the_runner_context():
    # GitHub rejects the whole workflow ("Unrecognized named-value: 'runner'") when a job-level `env` uses `runner.*`; step-level `env` and `with` may.
    offenders = []
    in_job_env = False
    for number, line in enumerate(LOCK.splitlines(), start=1):
        if line == "    env:":
            in_job_env = True
        elif in_job_env and (not line.strip() or len(line) - len(line.lstrip()) > 4):
            if "runner." in line and "${{" in line:
                offenders.append(number)
        else:
            in_job_env = False
    assert offenders == []


def step_position(name):
    return LOCK.index(f"name: {name}")


def test_knowledge_updates_run_in_the_expected_order_and_cannot_block_outputs():
    managed_upload = step_position("Upload repo-memory artifact (default)")
    snapshot = step_position("Snapshot trusted inputs for the knowledge base update")
    record = step_position("Record this run in the knowledge base")
    replace = step_position("Replace the repo-memory artifact with the trusted knowledge base")
    assert step_position("Clone repo-memory branch (default)") < step_position("Download nightly failure digest") < snapshot
    assert snapshot < step_position("Execute GitHub Copilot CLI")
    # The managed upload runs before post-steps, so the trusted result has to replace it.
    assert managed_upload < record < replace

    snapshot_step = LOCK[LOCK.rindex("      - ", 0, snapshot) : LOCK.index("\n      - ", snapshot)]
    assert "continue-on-error: true" in snapshot_step and '--trusted-dir \\"$RUNNER_TEMP/nightly-remediator-trusted\\"' in snapshot_step

    record_step = LOCK[LOCK.rindex("      - ", 0, record) : replace]
    assert "continue-on-error: true" in record_step and "if: always()" in record_step
    assert "GH_AW_SAFE_OUTPUTS" not in record_step and "safeoutputs" not in record_step
    # Runs the script copy taken before the agent, never code from the agent-writable workspace, and never reads the agent-writable memory clone.
    assert 'python3 -E -s \\"$trusted/scripts/update_knowledge_base.py\\" update' in record_step
    assert "GITHUB_WORKSPACE" not in record_step and ".github/scripts" not in record_step and "/tmp/gh-aw/repo-memory" not in record_step
    assert 'out=\\"$RUNNER_TEMP/nightly-remediator-memory\\"' in record_step and '--out-dir \\"$out\\"' in record_step
    # The placeholder that keeps the replacement artifact non-empty must be ignored by the push.
    placeholder = re.search(r'> \\"\$out/([^"\\]+)\\"', record_step).group(1)
    assert placeholder != kb.KNOWLEDGE_FILE and not placeholder.endswith(".json")
    replace_step = LOCK[LOCK.rindex("      - ", 0, replace) : LOCK.index("\n      # Small dedicated copy", replace)]
    assert "if: always()" in replace_step and "name: repo-memory-default" in replace_step and "overwrite: true" in replace_step
    assert "path: ${{ runner.temp }}/nightly-remediator-memory" in replace_step and "if-no-files-found: error" in replace_step


def test_workflow_steps_and_prompt_use_the_updater_cli_consistently():
    assert ukb.parse_args(["snapshot", "--digest-dir", "d", "--memory-dir", "m", "--trusted-dir", "t", "--notes-dir", "n"]).command == "snapshot"
    assert ukb.parse_args(["update", "--out-dir", "o", "--trusted-dir", "t", "--notes-dir", "n", "--remediator-run-url", "u"]).command == "update"
    for flag in ("--digest-dir", "--memory-dir", "--trusted-dir", "--notes-dir", "--out-dir", "--remediator-run-url"):
        assert flag in WORKFLOW
    notes_dir = "/tmp/gh-aw/agent/nightly-remediator/outcomes"
    assert f"--notes-dir {notes_dir}" in WORKFLOW and f"{notes_dir}/<12-hex-signature>.json" in WORKFLOW
    assert "/tmp/gh-aw/repo-memory/default/knowledge.json" in WORKFLOW
    assert "untrusted data" in WORKFLOW.split("## Knowledge base", 1)[1]
    for name in kb.NOTE_FIELDS:
        assert f'"{name}"' in WORKFLOW.split("**Writing.**", 1)[1]
    assert "jq -r" in WORKFLOW and shell_command_allowed(LOCK, "jq")
