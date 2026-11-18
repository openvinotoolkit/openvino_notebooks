"""GitHub CLI, Actions-run, and artifact access for the nightly collector."""

from __future__ import annotations

import json
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlencode

from collector_common import (
    ENVIRONMENTS_FILE,
    ESCAPE_SEQUENCES_FLAG,
    MAX_REPORT_BYTES,
    NIGHTLY_BRANCH,
    NIGHTLY_WORKFLOW_FILE,
    REPORT_ARTIFACT,
    REPORT_FILE,
    SUPPORTED_SCHEMA_MAJOR,
    CollectorError,
    GhError,
    NoWork,
    TRANSIENT_MARKERS,
)


def gh(args: List[str], stdout_path: Optional[Path] = None, retries: int = 3, timeout: int = 600) -> bytes:
    """Runs `gh`, retrying transient failures. With `stdout_path` the output is streamed to that file and b"" is returned."""
    for attempt in range(1, retries + 1):
        try:
            if stdout_path is None:
                proc = subprocess.run(["gh", *args], capture_output=True, timeout=timeout, check=False)
                output = proc.stdout
            else:
                with open(stdout_path, "wb") as handle:
                    proc = subprocess.run(["gh", *args], stdout=handle, stderr=subprocess.PIPE, timeout=timeout, check=False)
                output = b""
            if proc.returncode == 0:
                return output
            error = GhError(args, proc.returncode, proc.stderr.decode("utf-8", errors="replace"))
        except subprocess.TimeoutExpired:
            error = GhError(args, -1, f"timed out after {timeout}s")
        # Job logs contain ANSI codes, which recent gh versions refuse to print unless asked; older versions do not know the flag.
        if ESCAPE_SEQUENCES_FLAG in error.stderr and args[:1] == ["api"] and ESCAPE_SEQUENCES_FLAG not in args:
            return gh([args[0], ESCAPE_SEQUENCES_FLAG, *args[1:]], stdout_path, retries, timeout)
        if attempt == retries or not any(marker in error.stderr.lower() for marker in TRANSIENT_MARKERS):
            raise error
        time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def api_json(path: str) -> Any:
    return json.loads(gh(["api", path]))


def list_json(args: List[str]) -> Any:
    return json.loads(gh(args))


def download_artifact(source_repo: str, artifact_id: int, dest: Path) -> None:
    gh(["api", f"repos/{source_repo}/actions/artifacts/{artifact_id}/zip"], stdout_path=dest)
    if not zipfile.is_zipfile(dest):
        raise CollectorError(f"artifact {artifact_id} of {source_repo} is not a zip archive")


def read_zip_member(zip_path: Path, name: str, max_bytes: int) -> Optional[bytes]:
    """Reads one named member (never extracts the archive); None if it does not exist."""
    with zipfile.ZipFile(zip_path) as archive:
        try:
            info = archive.getinfo(name)
        except KeyError:
            return None
        if info.file_size > max_bytes:
            raise CollectorError(f"{name} is larger than {max_bytes} bytes")
        with archive.open(info) as handle:
            return handle.read(max_bytes + 1)[:max_bytes]


def list_nightly_runs(source_repo: str, per_page: int = 30) -> List[dict]:
    query = urlencode({"branch": NIGHTLY_BRANCH, "event": "schedule", "status": "completed", "per_page": per_page})
    return api_json(f"repos/{source_repo}/actions/workflows/{NIGHTLY_WORKFLOW_FILE}/runs?{query}").get("workflow_runs", [])


def resolve_run(source_repo: str, run_id: Optional[str]) -> dict:
    """The explicit run, or the latest completed scheduled nightly of the default branch."""
    if run_id:
        run = api_json(f"repos/{source_repo}/actions/runs/{run_id}")
        if NIGHTLY_WORKFLOW_FILE not in str(run.get("path", "")):
            raise CollectorError(f"run {run_id} of {source_repo} is not a {NIGHTLY_WORKFLOW_FILE} run (path: {run.get('path')})")
        if run.get("status") != "completed":
            raise NoWork(f"run {run_id} has not completed yet (status: {run.get('status')})")
        return run
    for run in list_nightly_runs(source_repo, per_page=15):
        if run.get("conclusion") in ("success", "failure"):
            return run
    raise NoWork(f"no completed scheduled {NIGHTLY_WORKFLOW_FILE} run found on {NIGHTLY_BRANCH}")


def find_artifact(source_repo: str, run_id: int, name: str) -> Optional[dict]:
    data = api_json(f"repos/{source_repo}/actions/runs/{run_id}/artifacts?{urlencode({'name': name, 'per_page': 100})}")
    candidates = [a for a in data.get("artifacts", []) if a.get("name") == name and not a.get("expired")]
    return max(candidates, key=lambda a: a["id"]) if candidates else None


def download_report(source_repo: str, run_id: int, workdir: Path, with_environments: bool) -> Tuple[dict, dict]:
    artifact = find_artifact(source_repo, run_id, REPORT_ARTIFACT)
    if artifact is None:
        raise NoWork(f"artifact {REPORT_ARTIFACT} of run {run_id} was not found (not produced or expired)")
    zip_path = workdir / f"report-{artifact['id']}.zip"
    download_artifact(source_repo, artifact["id"], zip_path)
    raw = read_zip_member(zip_path, REPORT_FILE, MAX_REPORT_BYTES)
    if raw is None:
        raise NoWork(f"{REPORT_FILE} is missing in artifact {REPORT_ARTIFACT} of run {run_id}")
    report = json.loads(raw)
    environments: dict = {}
    if with_environments:
        raw_envs = read_zip_member(zip_path, ENVIRONMENTS_FILE, MAX_REPORT_BYTES)
        environments = json.loads(raw_envs) if raw_envs else {}
    zip_path.unlink(missing_ok=True)
    return report, environments


def check_schema(report: dict) -> None:
    version = str(report.get("schema_version", ""))
    major = version.split(".", 1)[0]
    if not major.isdigit() or int(major) != SUPPORTED_SCHEMA_MAJOR:
        raise NoWork(f"unsupported nightly report schema version {version!r} (supported major version: {SUPPORTED_SCHEMA_MAJOR})")
    for key in ("run", "failures", "results", "jobs", "configs"):
        if key not in report:
            raise NoWork(f"nightly report has no `{key}` section")
