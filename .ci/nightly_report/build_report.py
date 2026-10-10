"""Build the nightly dashboard report from the artifacts of a treon_nightly workflow run.

Inputs:
  * plan.json written by split_notebooks.py (--plan_file): batch membership of notebooks;
  * fragment.json (+ logs/, envs/) of every test job written by validate_notebooks.py (--dashboard_dir);
  * GitHub REST API: run metadata, jobs (conclusions, failed steps, links) and artifacts (log links);
  * the repository checkout: notebook catalog (title, tags) and git metadata.

Outputs (--output_dir): nightly-report.json, environments.json, summary.md and a copy of the report schema.

Data problems (missing fragments, API errors, unparsable logs) never stop the build; they are recorded in
generator.warnings. The script exits with a non-zero code only if the produced report violates the schema.
"""

import argparse
import hashlib
import http.client
import itertools
import json
import os
import re
import shutil
import subprocess  # nosec B404 - runs git with fixed arguments
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nightly_report.constants import (  # noqa: E402
    ENVIRONMENTS_FILE_NAME,
    ENVS_DIR_NAME,
    FRAGMENT_ARTIFACT_PREFIX,
    FRAGMENT_FILE_NAME,
    REPORT_FILE_NAME,
    SCHEMA_VERSION,
    SUMMARY_FILE_NAME,
    NotRunReason,
    Status,
    config_id as make_config_id,
)
from nightly_report.error_parser import analyze_failure, compute_signature  # noqa: E402
from nightly_report.fragment import parse_pip_freeze  # noqa: E402
from nightly_report.metrics import compute_summary  # noqa: E402
from nightly_report.summary_md import render_markdown  # noqa: E402

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
REPORT_SCHEMA_PATH = SCHEMA_DIR / "nightly-report.schema.json"
FRAGMENT_SCHEMA_PATH = SCHEMA_DIR / "fragment.schema.json"

TEST_JOB_PREFIX = "build_treon"
TREON_STEP_PREFIX = "Analysing with treon"
JOB_MATRIX_RE = re.compile(r"^(?P<prefix>[\w-]+)\s*\((?P<matrix>[^)]*)\)")


class Warnings:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, message: str) -> None:
        print(f"WARNING: {message}", flush=True)
        self.items.append(message)


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def format_time(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def seconds_between(start: Optional[str], end: Optional[str]) -> Optional[float]:
    start_time, end_time = parse_time(start), parse_time(end)
    if start_time is None or end_time is None:
        return None
    return round(max((end_time - start_time).total_seconds(), 0.0), 1)


def os_family(os_name: str) -> str:
    lowered = os_name.lower()
    for prefix, family in (("ubuntu", "linux"), ("linux", "linux"), ("windows", "windows"), ("macos", "macos")):
        if lowered.startswith(prefix):
            return family
    return "unknown"


def runner_label_to_os_device(runner_label: str) -> tuple[str, str]:
    """Mirror of the 'Set env variables' step of job_unix.yml / job_windows.yml."""
    os_name = "ubuntu-22.04" if "linux" in runner_label else "macos-13" if "mac" in runner_label else "windows-2022"
    device = "gpu" if runner_label == "gpu" else "cpu"
    return os_name, device


def parse_job_name(name: str) -> Optional[dict]:
    """'build_treon_linux (aks-linux-8-cores-32gb, 3.11, batch_2) / build_treon' -> matrix values."""
    match = JOB_MATRIX_RE.match(name)
    if not match or not match.group("prefix").startswith(TEST_JOB_PREFIX):
        return None
    values = [value.strip() for value in match.group("matrix").split(",") if value.strip()]
    if not values:
        return None
    python = next((value for value in values if re.fullmatch(r"\d+\.\d+", value)), None)
    batch = next((value for value in values if re.fullmatch(r"batch_\d+", value)), None)
    if python is None:
        return None
    runner_label = values[0]
    os_name, device = runner_label_to_os_device(runner_label)
    return {"runner_label": runner_label, "python": python, "batch": batch, "os": os_name, "device": device}


class GitHubApi:
    def __init__(self, api_url: str, repository: str, token: Optional[str], warnings: Warnings) -> None:
        self.api_url = api_url.rstrip("/")
        self.repository = repository
        self.token = token
        self.warnings = warnings

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        url = f"{self.api_url}/repos/{self.repository}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310 - fixed https API URL
                    return json.loads(response.read().decode("utf-8"))
            # OSError covers URLError, timeouts and connection resets; HTTPException covers e.g. RemoteDisconnected, IncompleteRead.
            except (OSError, http.client.HTTPException, ValueError) as error:
                last_error = error
                if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code != 429:
                    break
                if attempt < 2:
                    time.sleep(2**attempt)
        self.warnings.add(f"GitHub API request failed: {url}: {last_error}")
        return None

    def get_run(self, run_id: int) -> Optional[dict]:
        return self._get(f"actions/runs/{run_id}")

    def _paginate(self, path: str, key: str, params: dict) -> Optional[list]:
        items: list = []
        page = 1
        while True:
            data = self._get(path, {**params, "per_page": 100, "page": page})
            if data is None:
                return items or None
            batch = data.get(key, [])
            items.extend(batch)
            if len(batch) < 100 or len(items) >= data.get("total_count", 0):
                return items
            page += 1

    def get_jobs(self, run_id: int) -> Optional[list]:
        return self._paginate(f"actions/runs/{run_id}/jobs", "jobs", {"filter": "latest"})

    def get_artifacts(self, run_id: int) -> Optional[list]:
        return self._paginate(f"actions/runs/{run_id}/artifacts", "artifacts", {})


def run_git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), *args], stderr=subprocess.DEVNULL, text=True, encoding="utf-8")  # nosec B603 B607
    except (OSError, subprocess.CalledProcessError):
        return None


def get_tree_shas(repo_root: Path, notebooks_dir: str) -> dict[str, str]:
    output = run_git(repo_root, "ls-tree", "-r", "-d", "HEAD", "--", f"{notebooks_dir}/")
    tree_shas: dict[str, str] = {}
    for line in (output or "").splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "tree" and path.startswith(f"{notebooks_dir}/"):
            tree_shas[path[len(notebooks_dir) + 1 :]] = parts[2]
    return tree_shas


def get_last_modified(repo_root: Path, notebooks_dir: str, dirs: Iterable[str], warnings: Warnings) -> dict[str, dict]:
    """Latest commit touching each notebook directory, found in a single pass over the history."""
    pending = set(dirs)
    if not pending:
        return {}
    if (run_git(repo_root, "rev-parse", "--is-shallow-repository") or "").strip() != "false":
        warnings.add("Repository checkout is shallow or not a git repository; notebook last_modified is not collected.")
        return {}
    command = ["git", "-C", str(repo_root), "log", "--format=%x1e%H%x1f%cI%x1f%an%x1f%s", "--name-only", "HEAD", "--", f"{notebooks_dir}/"]
    found: dict[str, dict] = {}
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")  # nosec B603
    except OSError as error:
        warnings.add(f"git log failed: {error}")
        return {}
    commit: Optional[dict] = None
    try:
        for line in process.stdout:
            line = line.rstrip("\n")
            if line.startswith("\x1e"):
                sha, date, author, subject = (line[1:].split("\x1f") + ["", "", "", ""])[:4]
                commit = {"sha": sha, "date": date, "author": author, "message": subject}
                continue
            if not line or commit is None or not line.startswith(f"{notebooks_dir}/"):
                continue
            parts = line[len(notebooks_dir) + 1 :].split("/")[:-1]
            for depth in range(1, len(parts) + 1):
                directory = "/".join(parts[:depth])
                if directory in pending:
                    pending.discard(directory)
                    found[directory] = commit
            if not pending:
                break
    finally:
        process.kill()
        process.wait()
    return found


def get_head_commit(repo_root: Path) -> Optional[dict]:
    output = run_git(repo_root, "log", "-1", "--format=%H%x1f%cI%x1f%an%x1f%s")
    if not output:
        return None
    sha, date, author, subject = (output.strip().split("\x1f") + ["", "", "", ""])[:4]
    return {"sha": sha, "date": date, "author": author, "message": subject}


def read_notebook_info(path: Path) -> tuple[str, dict]:
    title, tags = "", {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return title, tags
    cells = content.get("cells") or []
    if cells:
        source = cells[0].get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        match = re.search(r"# (.+)", source)
        if match:
            title = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", match.group(1)).strip()
    raw_tags = (content.get("metadata") or {}).get("openvino_notebooks", {}).get("tags") or {}
    if isinstance(raw_tags, dict):
        tags = {key: [str(item) for item in value] for key, value in raw_tags.items() if isinstance(value, list)}
    return title, tags


def commit_entry(commit: Optional[dict], links: "Links") -> Optional[dict]:
    if not commit or not commit.get("sha"):
        return None
    sha = commit["sha"]
    return {
        "sha": sha,
        "short_sha": sha[:7],
        "message": (commit.get("message") or "").splitlines()[0] if commit.get("message") else None,
        "author": commit.get("author") or None,
        "date": format_time(parse_time(commit.get("date"))),
        "url": links.commit(sha),
    }


class Links:
    def __init__(self, server_url: str, repository: str, run_id: int) -> None:
        self.base = f"{server_url.rstrip('/')}/{repository}"
        self.run_id = run_id

    def run(self) -> str:
        return f"{self.base}/actions/runs/{self.run_id}"

    def job(self, job_id: int) -> str:
        return f"{self.run()}/job/{job_id}"

    def step(self, job_url: str, number: int) -> str:
        return f"{job_url}#step:{number}:1"

    def artifact(self, artifact_id: int) -> str:
        return f"{self.run()}/artifacts/{artifact_id}"

    def commit(self, sha: str) -> str:
        return f"{self.base}/commit/{sha}"

    def blob(self, ref: str, path: str) -> str:
        return f"{self.base}/blob/{ref}/{path}"

    def history(self, ref: str, path: str) -> str:
        return f"{self.base}/commits/{ref}/{path}"


def load_json(path: Optional[Path], warnings: Warnings, what: str) -> Optional[dict]:
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        warnings.add(f"Cannot read {what} '{path}': {error}")
        return None


def get_validator(schema_path: Path):
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return None
    return Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))


def schema_errors(validator, data: dict, limit: int = 20) -> list[str]:
    if validator is None:
        return []
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    return [f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}" for error in errors[:limit]]


def load_fragments(fragments_dir: Path, warnings: Warnings) -> list[dict]:
    validator = get_validator(FRAGMENT_SCHEMA_PATH)
    fragments = []
    if not fragments_dir.exists():
        warnings.add(f"Fragments directory '{fragments_dir}' does not exist.")
        return fragments
    for path in sorted(fragments_dir.rglob(FRAGMENT_FILE_NAME)):
        data = load_json(path, warnings, "fragment")
        if not isinstance(data, dict) or not isinstance(data.get("job"), dict):
            continue
        for error in schema_errors(validator, data, limit=5):
            warnings.add(f"Fragment '{path}' does not match the schema: {error}")
        job = data["job"]
        try:
            data["_config_id"] = make_config_id(job["device"], job["os"], job["python"])
        except KeyError:
            warnings.add(f"Fragment '{path}' has no device/os/python; ignored.")
            continue
        data["_dir"] = path.parent
        fragments.append(data)
    return fragments


def first_line(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return text.strip().splitlines()[0] if text.strip() else None


class ReportBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.warnings = Warnings()
        self.repo_root: Path = args.repo_root.resolve()
        self.notebooks_dir: str = args.notebooks_dir.strip("/")
        self.repository: str = args.repository
        self.run_id: int = args.run_id
        self.links = Links(args.server_url, self.repository, self.run_id)
        self.api = (
            None if args.no_api else GitHubApi(args.api_url, self.repository, os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"), self.warnings)
        )

    # ---------------------------------------------------------------- inputs
    def _api_or_file(self, file_path: Optional[Path], key: Optional[str], fetch) -> Optional[object]:
        if file_path is not None:
            data = load_json(file_path, self.warnings, "API mock")
            if data is None:
                return None
            return data.get(key, []) if key else data
        if self.api is None:
            return None
        return fetch()

    def load_inputs(self) -> None:
        self.run_data = self._api_or_file(self.args.run_json, None, lambda: self.api.get_run(self.run_id)) or {}
        self.api_jobs = self._api_or_file(self.args.jobs_json, "jobs", lambda: self.api.get_jobs(self.run_id))
        if self.api_jobs is None:
            self.warnings.add("Job information is unavailable; job conclusions and links are incomplete.")
            self.api_jobs = []
        self.api_artifacts = self._api_or_file(self.args.artifacts_json, "artifacts", lambda: self.api.get_artifacts(self.run_id)) or []
        self.plan = load_json(self.args.plan, self.warnings, "plan") if self.args.plan else None
        if self.plan is None:
            self.warnings.add("Batch plan is unavailable; notebooks of jobs without fragments cannot be reported.")
        self.fragments = load_fragments(self.args.fragments_dir, self.warnings)
        self.head_sha = self.run_data.get("head_sha") or os.environ.get("GITHUB_SHA") or (get_head_commit(self.repo_root) or {}).get("sha") or "unknown"
        timeouts = {fragment["job"].get("timeout_s") for fragment in self.fragments if fragment["job"].get("timeout_s") is not None}
        self.timeout_s = self.args.timeout if self.args.timeout is not None else (timeouts.pop() if len(timeouts) == 1 else None)

    # ---------------------------------------------------------------- jobs
    def _load_workflow(self) -> tuple[dict[str, Optional[list[dict]]], set[str]]:
        """Reads the workflow file once: test job id -> expanded matrix (None for jobs disabled with a constant false `if`)
        and the ids of the jobs the test jobs depend on (`needs`)."""
        if getattr(self, "_workflow", None) is not None:
            return self._workflow
        self._workflow = ({}, set())
        if not self.args.workflow_file:
            return self._workflow
        try:
            import yaml

            workflow = yaml.safe_load((self.repo_root / self.args.workflow_file).read_text(encoding="utf-8")) or {}
        except Exception as error:
            self.warnings.add(f"Cannot read workflow file '{self.args.workflow_file}': {error}")
            return self._workflow
        matrices, upstream = self._workflow
        for job_key, job in (workflow.get("jobs") or {}).items():
            if not str(job_key).startswith(TEST_JOB_PREFIX) or not isinstance(job, dict):
                continue
            if str(job.get("if", "")).replace(" ", "") in ("False", "false", "${{false}}"):
                matrices[job_key] = None
                continue
            needs = job.get("needs") or []
            upstream.update([needs] if isinstance(needs, str) else needs)
            matrix = (job.get("strategy") or {}).get("matrix") or {}
            runners, pythons, batches = matrix.get("runs_on") or [], matrix.get("python") or [], matrix.get("batch") or [None]
            combinations = []
            for runner_label, python, batch in itertools.product(runners, pythons, batches):
                os_name, device = runner_label_to_os_device(str(runner_label))
                combinations.append({"runner_label": str(runner_label), "python": str(python), "batch": batch, "os": os_name, "device": device})
            matrices[job_key] = combinations
        return self._workflow

    def _workflow_matrices(self) -> dict[str, Optional[list[dict]]]:
        return self._load_workflow()[0]

    def _upstream_jobs(self) -> set[str]:
        return self._load_workflow()[1]

    def collect_jobs(self) -> list[dict]:
        """Join API jobs with fragments; returns internal job records (with private keys)."""
        artifact_ids = {artifact.get("name"): artifact.get("id") for artifact in self.api_artifacts if not artifact.get("expired")}
        fragments_by_job_id = {fragment["job"].get("job_id"): fragment for fragment in self.fragments if fragment["job"].get("job_id")}
        fragments_by_key = {(fragment["_config_id"], fragment["job"].get("batch")): fragment for fragment in self.fragments}
        used_fragments: set[int] = set()
        records = []

        for api_job in self.api_jobs:
            name = api_job.get("name") or ""
            matrix = parse_job_name(name)
            fragment = fragments_by_job_id.get(api_job.get("id"))
            if fragment is None and matrix is None and name.startswith(TEST_JOB_PREFIX):
                # Jobs skipped or cancelled before start are not expanded by GitHub: one job without matrix values in its name.
                combinations = self._workflow_matrices().get(name.split(" ")[0], [])
                if combinations is None:
                    continue
                if not combinations:
                    self.warnings.add(f"Test job '{name}' ({api_job.get('conclusion')}) has no expanded matrix; its notebooks are not reported.")
                    continue
                for combination in combinations:
                    record = self._job_record(api_job, combination, None, artifact_ids)
                    record["name"] = f"{name} ({combination['runner_label']}, {combination['python']}, {combination['batch']})"
                    records.append(record)
                continue
            if fragment is None and matrix is not None:
                candidate = fragments_by_key.get((make_config_id(matrix["device"], matrix["os"], matrix["python"]), matrix["batch"]))
                if candidate is not None and self._is_stale(candidate, api_job):
                    self.warnings.add(
                        f"Fragment of {candidate['_config_id']}/{candidate['job'].get('batch')} comes from job {candidate['job'].get('job_id')} "
                        f"(attempt {candidate['job'].get('run_attempt')}), not from the latest job {api_job.get('id')} "
                        f"(attempt {api_job.get('run_attempt')}); it is ignored."
                    )
                    used_fragments.add(id(candidate))
                    candidate = None
                fragment = candidate
            if matrix is None and fragment is None:
                continue
            records.append(self._job_record(api_job, matrix, fragment, artifact_ids))
            if fragment is not None:
                used_fragments.add(id(fragment))

        for fragment in self.fragments:
            if id(fragment) in used_fragments:
                continue
            if self.api_jobs:
                self.warnings.add(f"Fragment of {fragment['_config_id']}/{fragment['job'].get('batch')} does not match any job of the run; it is ignored.")
            else:
                records.append(self._job_record(None, None, fragment, artifact_ids))
        records.sort(key=lambda record: (record["config_id"], record["batch"] or "", record["id"]))
        return records

    @staticmethod
    def _is_stale(fragment: dict, api_job: dict) -> bool:
        """A fragment written by another job (e.g. an earlier attempt) than the latest one of its config and batch."""
        fragment_job_id, fragment_attempt = fragment["job"].get("job_id"), fragment["job"].get("run_attempt")
        if fragment_job_id is not None and api_job.get("id") is not None:
            return fragment_job_id != api_job.get("id")
        return fragment_attempt is not None and api_job.get("run_attempt") is not None and fragment_attempt != api_job.get("run_attempt")

    def _job_record(self, api_job: Optional[dict], matrix: Optional[dict], fragment: Optional[dict], artifact_ids: dict) -> dict:
        api_job = api_job or {}
        fragment_job = fragment["job"] if fragment else {}
        device = fragment_job.get("device") or matrix["device"]
        os_name = fragment_job.get("os") or matrix["os"]
        python = fragment_job.get("python") or matrix["python"]
        batch = fragment_job.get("batch") if fragment else matrix["batch"]
        cfg_id = make_config_id(device, os_name, python)
        job_id = api_job.get("id") or fragment_job.get("job_id") or 0
        job_url = api_job.get("html_url") or self.links.job(job_id)

        steps = api_job.get("steps") or []
        treon_step = next(
            (step for step in steps if str(step.get("name", "")).startswith(TREON_STEP_PREFIX) and step.get("conclusion") not in (None, "skipped")), None
        )
        failed_step = next((step for step in steps if step.get("conclusion") == "failure"), None)
        conclusion = api_job.get("conclusion") or "unknown"
        tests_started = treon_step is not None if steps else fragment is not None

        artifact_name = f"{FRAGMENT_ARTIFACT_PREFIX}{device}-{os_name}-{python}-{batch or ''}"
        artifact_id = artifact_ids.get(artifact_name)
        return {
            "id": int(job_id),
            "name": api_job.get("name") or f"{cfg_id} {batch or ''}".strip(),
            "config_id": cfg_id,
            "batch": batch,
            "conclusion": conclusion,
            "started_at": format_time(parse_time(api_job.get("started_at"))) or fragment_job.get("started_at"),
            "completed_at": format_time(parse_time(api_job.get("completed_at"))) or fragment_job.get("finished_at") or fragment_job.get("updated_at"),
            "duration_s": None,
            "runner_name": api_job.get("runner_name") or fragment_job.get("runner_name"),
            "python_full_version": fragment_job.get("python_full_version"),
            "url": job_url,
            "treon_step_url": self.links.step(job_url, treon_step["number"]) if treon_step and treon_step.get("number") else None,
            "logs_artifact_url": self.links.artifact(artifact_id) if artifact_id else None,
            "failed_step": (
                {"number": failed_step["number"], "name": failed_step.get("name", ""), "url": self.links.step(job_url, failed_step["number"])}
                if failed_step and failed_step.get("number") is not None
                else None
            ),
            "tests_started": tests_started,
            "fragment_present": fragment is not None,
            "notebooks_planned": len(fragment["planned"]) if fragment else None,
            "notebooks_completed": len(fragment["results"]) if fragment else None,
            "_device": device,
            "_os": os_name,
            "_python": python,
            "_runner_label": fragment_job.get("runner_label") or (matrix or {}).get("runner_label"),
            "_container_image": fragment_job.get("container_image"),
            "_fragment": fragment,
        }

    # ---------------------------------------------------------------- results
    def _plan_batch(self, batch: Optional[str]) -> Optional[list[str]]:
        if not self.plan or not batch:
            return None
        notebooks = (self.plan.get("batches") or {}).get(batch)
        if notebooks is None:
            return None
        prefix = f"{self.plan.get('notebooks_dir', self.notebooks_dir).strip('/')}/"
        return [path[len(prefix) :] if path.startswith(prefix) else path for path in notebooks]

    def _skip_reasons(self, job: dict) -> dict[str, str]:
        try:
            from skip_resolution import resolve_skip_reasons
        except ImportError as error:
            self.warnings.add(f"Skip resolution is unavailable ({error}); skipped notebooks of crashed jobs are reported as not_run.")
            return {}
        validation_config = {"os": job["_os"], "python": job["_python"], "device": job["_device"]}
        ignore_lists = [str(self.repo_root / item) if item.endswith(".txt") else item for item in self.args.ignore_list]
        try:
            reasons = resolve_skip_reasons(validation_config, self.repo_root / self.args.skip_config, ignore_lists)
        except Exception as error:
            self.warnings.add(f"Skip resolution failed for {job['config_id']}: {error}")
            return {}
        prefix = f"{self.notebooks_dir}/"
        return {(path[len(prefix) :] if path.startswith(prefix) else path): reason for path, reason in reasons.items()}

    def _not_run_reason(self, job: dict) -> str:
        if job["conclusion"] == "skipped":
            return NotRunReason.JOB_SKIPPED
        if job["tests_started"]:
            return NotRunReason.JOB_CANCELLED if job["conclusion"] == "cancelled" else NotRunReason.NO_FRAGMENT
        if job["failed_step"]:
            return f"{NotRunReason.JOB_FAILED_BEFORE_TESTS}:{job['failed_step']['name']}"
        if job["conclusion"] == "cancelled":
            return NotRunReason.JOB_CANCELLED
        return NotRunReason.JOB_FAILED_BEFORE_TESTS

    def collect_results(self, jobs: list[dict]) -> tuple[list[dict], dict[str, tuple[dict, dict]]]:
        """Returns result rows and failure_id -> (fragment result, job) for later failure analysis."""
        results: list[dict] = []
        failing: dict[str, tuple[dict, dict]] = {}
        seen: set[tuple[str, str]] = set()

        def add(row: dict) -> None:
            key = (row["notebook"], row["config_id"])
            if key in seen:
                self.warnings.add(f"Duplicate result for {row['config_id']}/{row['notebook']} ignored.")
                return
            seen.add(key)
            results.append(row)

        for job in jobs:
            base = {"config_id": job["config_id"], "batch": job["batch"], "job_id": job["id"]}
            fragment = job["_fragment"]
            if fragment is not None:
                timeout_s = fragment["job"].get("timeout_s")
                by_notebook = {result["notebook"]: result for result in fragment.get("results", []) if isinstance(result, dict) and "notebook" in result}
                planned = list(fragment.get("planned") or [])
                planned += [notebook for notebook in by_notebook if notebook not in planned]
                interrupted_reason = NotRunReason.JOB_CANCELLED if job["conclusion"] == "cancelled" else NotRunReason.JOB_INTERRUPTED
                for notebook in planned:
                    result = by_notebook.get(notebook)
                    if result is None:
                        add(self._row(base, notebook, Status.NOT_RUN, timeout_s=timeout_s, not_run_reason=interrupted_reason))
                        continue
                    status = result.get("status") if result.get("status") in Status.ALL else Status.ERROR
                    row = self._row(base, notebook, status, timeout_s=timeout_s, result=result)
                    if status in Status.FAILING:
                        row["failure_id"] = f"{job['config_id']}/{notebook}"
                        failing[row["failure_id"]] = (result, job)
                    add(row)
                continue

            planned = self._plan_batch(job["batch"])
            if planned is None:
                self.warnings.add(f"No fragment and no plan for job '{job['name']}'; its notebooks are not reported.")
                continue
            skip_reasons = self._skip_reasons(job)
            reason = self._not_run_reason(job)
            for notebook in planned:
                if notebook in skip_reasons:
                    add(self._row(base, notebook, Status.SKIPPED, timeout_s=self.timeout_s, skip_reason=skip_reasons[notebook]))
                else:
                    add(self._row(base, notebook, Status.NOT_RUN, timeout_s=self.timeout_s, not_run_reason=reason))
        results.sort(key=lambda row: (row["notebook"], row["config_id"]))
        return results, failing

    @staticmethod
    def _row(base: dict, notebook: str, status: str, timeout_s=None, result: Optional[dict] = None, skip_reason=None, not_run_reason=None) -> dict:
        result = result or {}
        duration = result.get("duration_s")
        return {
            "notebook": notebook,
            **base,
            "status": status,
            "skip_reason": result.get("skip_reason", skip_reason) if status == Status.SKIPPED else None,
            "not_run_reason": not_run_reason,
            "started_at": result.get("started_at"),
            "duration_s": round(duration, 2) if isinstance(duration, (int, float)) else None,
            "timeout_s": timeout_s,
            "peak_rss_bytes": result.get("peak_rss_bytes"),
            "openvino_before": result.get("openvino_before"),
            "packages": dict(result.get("packages") or {}),
            "env_id": result.get("env_id"),
            "failure_id": None,
        }

    # ---------------------------------------------------------------- failures
    def collect_failures(self, failing: dict[str, tuple[dict, dict]], results: list[dict]) -> list[dict]:
        failures = []
        for row in results:
            if not row["failure_id"]:
                continue
            result, job = failing[row["failure_id"]]
            text = None
            log_file = result.get("log_file")
            if log_file:
                log_path = Path(job["_fragment"]["_dir"]) / log_file
                try:
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                except OSError as error:
                    self.warnings.add(f"Cannot read log '{log_path}': {error}")
            try:
                analysis = analyze_failure(text, row["status"], result.get("return_code"), result.get("error_hint"))
            except Exception as error:  # the parser must never break the report
                self.warnings.add(f"Log analysis failed for {row['failure_id']}: {error}")
                category = "timeout" if row["status"] == Status.TIMEOUT else "harness_error" if row["status"] == Status.ERROR else "unknown"
                analysis = {"category": category, "error_class": "other", "signature": compute_signature(category, None, None)}
            failures.append(
                {
                    "id": row["failure_id"],
                    "notebook": row["notebook"],
                    "config_id": row["config_id"],
                    "job_id": row["job_id"],
                    "status": row["status"],
                    "category": analysis.get("category", "unknown"),
                    "error_class": analysis.get("error_class", "other"),
                    "signature": analysis.get("signature") or hashlib.sha1(row["failure_id"].encode()).hexdigest()[:12],  # nosec B324
                    "cell_index": analysis.get("cell_index"),
                    "cell_source_excerpt": analysis.get("cell_source_excerpt"),
                    "exception_type": analysis.get("exception_type"),
                    "exception_message": analysis.get("exception_message"),
                    "traceback_excerpt": analysis.get("traceback_excerpt"),
                    "log_tail": analysis.get("log_tail"),
                    "links": {
                        "job": job["url"],
                        "treon_step": job["treon_step_url"],
                        "logs_artifact": job["logs_artifact_url"],
                        "log_file": log_file,
                        "notebook_source": self.links.blob(self.head_sha, f"{self.notebooks_dir}/{row['notebook']}"),
                    },
                }
            )
        return failures

    @staticmethod
    def group_errors(failures: list[dict]) -> list[dict]:
        groups: dict[str, dict] = {}
        for failure in sorted(failures, key=lambda item: (item["notebook"], item["config_id"])):
            group = groups.setdefault(
                failure["signature"],
                {
                    "signature": failure["signature"],
                    "category": failure["category"],
                    "error_class": failure["error_class"],
                    "exception_type": failure["exception_type"],
                    "exception_message": first_line(failure["exception_message"]),
                    "occurrences": 0,
                    "notebooks": [],
                    "configs": [],
                },
            )
            group["occurrences"] += 1
            if failure["notebook"] not in group["notebooks"]:
                group["notebooks"].append(failure["notebook"])
            if failure["config_id"] not in group["configs"]:
                group["configs"].append(failure["config_id"])
        for group in groups.values():
            group["notebooks"].sort()
            group["configs"].sort()
        return sorted(groups.values(), key=lambda group: (-group["occurrences"], group["signature"]))

    # ---------------------------------------------------------------- catalog
    def collect_notebooks(self, results: list[dict]) -> list[dict]:
        notebooks_root = self.repo_root / self.notebooks_dir
        paths = set()
        if notebooks_root.exists():
            for path in notebooks_root.rglob("*.ipynb"):
                relative = path.relative_to(notebooks_root)
                if path.name.startswith("test_") or ".ipynb_checkpoints" in relative.parts:
                    continue
                paths.add(relative.as_posix())
        else:
            self.warnings.add(f"Notebooks directory '{notebooks_root}' does not exist.")
        missing = {row["notebook"] for row in results} - paths
        if missing:
            self.warnings.add(f"{len(missing)} notebook(s) with results are missing in the checkout: {sorted(missing)[:10]}")
        paths |= missing

        dirs = {str(Path(path).parent.as_posix()) for path in paths}
        tree_shas = get_tree_shas(self.repo_root, self.notebooks_dir)
        last_modified = get_last_modified(self.repo_root, self.notebooks_dir, dirs, self.warnings)
        catalog = []
        for path in sorted(paths):
            directory = Path(path).parent.as_posix()
            title, tags = read_notebook_info(notebooks_root / path)
            catalog.append(
                {
                    "path": path,
                    "dir": directory,
                    "title": title or Path(path).stem,
                    "tags": tags,
                    "dir_tree_sha": tree_shas.get(directory),
                    "last_modified": commit_entry(last_modified.get(directory), self.links),
                    "links": {
                        "source": self.links.blob(self.head_sha, f"{self.notebooks_dir}/{path}"),
                        "latest": self.links.blob(self.args.latest_ref, f"{self.notebooks_dir}/{path}") if self.args.latest_ref else None,
                        "history": self.links.history(self.head_sha, f"{self.notebooks_dir}/{directory}"),
                    },
                }
            )
        return catalog

    # ---------------------------------------------------------------- run
    def build_run(self, jobs: list[dict]) -> dict:
        run = self.run_data
        created_at = format_time(parse_time(run.get("created_at")))
        started_at = format_time(parse_time(run.get("run_started_at"))) or created_at
        completed = [parse_time(job["completed_at"]) for job in jobs if job["completed_at"]]
        finished_at = format_time(max(completed)) if completed else None
        if started_at is None:
            starts = [parse_time(job["started_at"]) for job in jobs if job["started_at"]]
            started_at = format_time(min(starts)) if starts else None
        nightly_date = (parse_time(created_at or started_at) or datetime.now(timezone.utc)).date().isoformat()

        conclusions = [job["conclusion"] for job in jobs]
        # Jobs the test jobs depend on (e.g. collect_notebooks) also decide the outcome; jobs running in parallel with this one
        # (e.g. other report jobs) are ignored as their conclusion depends on timing.
        upstream = self._upstream_jobs()
        conclusions += [
            api_job["conclusion"] for api_job in self.api_jobs if api_job.get("conclusion") and str(api_job.get("name", "")).split(" ")[0] in upstream
        ]
        if any(conclusion in ("failure", "timed_out") for conclusion in conclusions):
            conclusion = "failure"
        elif "cancelled" in conclusions:
            conclusion = "cancelled"
        elif "success" in conclusions and all(conclusion in ("success", "skipped") for conclusion in conclusions):
            conclusion = "success"
        else:
            conclusion = "unknown"

        head_commit = run.get("head_commit") or {}
        commit = None
        if head_commit.get("id"):
            commit = {
                "sha": head_commit["id"],
                "date": head_commit.get("timestamp"),
                "author": (head_commit.get("author") or {}).get("name"),
                "message": head_commit.get("message"),
            }
        if commit is None or commit["sha"] != self.head_sha:
            git_commit = get_head_commit(self.repo_root)
            commit = git_commit if git_commit and git_commit["sha"] == self.head_sha else {"sha": self.head_sha}

        return {
            "id": self.run_id,
            "attempt": int(run.get("run_attempt") or self.args.run_attempt or 1),
            "number": run.get("run_number") or (int(os.environ["GITHUB_RUN_NUMBER"]) if os.environ.get("GITHUB_RUN_NUMBER", "").isdigit() else None),
            "workflow": run.get("name") or os.environ.get("GITHUB_WORKFLOW"),
            "repository": self.repository,
            "event": run.get("event") or os.environ.get("GITHUB_EVENT_NAME") or "unknown",
            "branch": run.get("head_branch") or os.environ.get("GITHUB_REF_NAME") or "unknown",
            "nightly_date": nightly_date,
            "created_at": created_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_clock_s": seconds_between(started_at, finished_at),
            "conclusion": conclusion,
            "url": run.get("html_url") or self.links.run(),
            "commit": commit_entry(commit, self.links) or {"sha": self.head_sha},
            "inputs": {
                "ov_branch": self.args.ov_branch,
                "docker_tag": self.args.docker_tag,
                "timeout_s": self.timeout_s,
                "num_batches": (self.plan or {}).get("num_batches"),
                "ignore_lists": list(self.args.ignore_list),
            },
        }

    @staticmethod
    def build_configs(jobs: list[dict]) -> list[dict]:
        configs: dict[str, dict] = {}
        for job in jobs:
            config = configs.setdefault(
                job["config_id"],
                {
                    "id": job["config_id"],
                    "device": job["_device"],
                    "os": job["_os"],
                    "os_family": os_family(job["_os"]),
                    "python": job["_python"],
                    "runner_label": job["_runner_label"],
                    "container_image": job["_container_image"],
                    "job_ids": [],
                },
            )
            config["runner_label"] = config["runner_label"] or job["_runner_label"]
            config["container_image"] = config["container_image"] or job["_container_image"]
            config["job_ids"].append(job["id"])
        for config in configs.values():
            config["job_ids"] = sorted(set(config["job_ids"]))
        return [configs[key] for key in sorted(configs)]

    def build_environments(self, jobs: list[dict], results: list[dict]) -> dict:
        wanted = {row["env_id"] for row in results if row["env_id"]}
        environments: dict[str, dict] = {}
        for job in jobs:
            fragment = job["_fragment"]
            if fragment is None:
                continue
            envs_dir = Path(fragment["_dir"]) / ENVS_DIR_NAME
            for env_id in sorted(wanted - environments.keys()):
                env_file = envs_dir / f"{env_id}.txt"
                if env_file.exists():
                    environments[env_id] = parse_pip_freeze(env_file.read_text(encoding="utf-8", errors="replace"))
        missing = wanted - environments.keys()
        if missing:
            self.warnings.add(f"{len(missing)} environment(s) referenced by results are missing in fragments.")
        return {key: environments[key] for key in sorted(environments)}

    # ---------------------------------------------------------------- main
    def build(self) -> tuple[dict, dict]:
        self.load_inputs()
        jobs = self.collect_jobs()
        results, failing = self.collect_results(jobs)
        failures = self.collect_failures(failing, results)
        for job in jobs:
            job["duration_s"] = seconds_between(job["started_at"], job["completed_at"])
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": format_time(datetime.now(timezone.utc)),
            "generator": {"script": ".ci/nightly_report/build_report.py", "warnings": self.warnings.items},
            "run": self.build_run(jobs),
            "configs": self.build_configs(jobs),
            "jobs": [{key: value for key, value in job.items() if not key.startswith("_")} for job in jobs],
            "notebooks": self.collect_notebooks(results),
            "results": results,
            "failures": failures,
            "error_groups": self.group_errors(failures),
        }
        environments = self.build_environments(jobs, results)
        report["summary"] = compute_summary(report)
        return report, environments


def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fragments_dir", type=Path, required=True, help="Directory with downloaded nightly-fragment-* artifacts.")
    parser.add_argument("--plan", type=Path, help="plan.json written by split_notebooks.py --plan_file.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--notebooks_dir", default="notebooks", help="Notebooks directory relative to the repository root.")
    parser.add_argument("--skip_config", default=".ci/skipped_notebooks.yml", help="Skip config relative to the repository root.")
    parser.add_argument("--ignore_list", nargs="*", default=[], help="Ignore lists passed to validate_notebooks.py (repo-root relative).")
    parser.add_argument("--ov_branch")
    parser.add_argument("--docker_tag")
    parser.add_argument("--timeout", type=float, help="Per-notebook timeout in seconds.")
    parser.add_argument("--latest_ref", default="latest", help="Branch used for 'latest' notebook links.")
    parser.add_argument(
        "--workflow_file",
        help="Workflow file (repo-root relative) with the test job matrices; used to report jobs skipped or cancelled before matrix expansion.",
    )
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", "openvinotoolkit/openvino_notebooks"))
    parser.add_argument("--run_id", type=int, default=int(os.environ.get("GITHUB_RUN_ID", "0") or 0))
    parser.add_argument("--run_attempt", type=int, default=int(os.environ.get("GITHUB_RUN_ATTEMPT", "1") or 1))
    parser.add_argument("--server_url", default=os.environ.get("GITHUB_SERVER_URL", "https://github.com"))
    parser.add_argument("--api_url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    parser.add_argument("--no_api", action="store_true", help="Do not call the GitHub API.")
    parser.add_argument("--run_json", type=Path, help="Use this file instead of GET /actions/runs/{id}.")
    parser.add_argument("--jobs_json", type=Path, help="Use this file ({'jobs': [...]}) instead of the jobs API.")
    parser.add_argument("--artifacts_json", type=Path, help="Use this file ({'artifacts': [...]}) instead of the artifacts API.")
    parser.add_argument("--summary_file", type=Path, help="Append the markdown summary to this file (e.g. $GITHUB_STEP_SUMMARY).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_arguments(argv)
    builder = ReportBuilder(args)
    report, environments = builder.build()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / REPORT_FILE_NAME).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / ENVIRONMENTS_FILE_NAME).write_text(json.dumps(environments, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copyfile(REPORT_SCHEMA_PATH, args.output_dir / REPORT_SCHEMA_PATH.name)

    markdown = render_markdown(report)
    (args.output_dir / SUMMARY_FILE_NAME).write_text(markdown, encoding="utf-8")
    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as summary_file:
            summary_file.write(markdown + "\n")

    summary = report["summary"]
    print(
        f"Report written to '{args.output_dir / REPORT_FILE_NAME}': {len(report['results'])} results, {len(report['failures'])} failures, "
        f"pass rate {summary['pass_rate']}, coverage {summary['coverage']}, {len(builder.warnings.items)} warning(s)."
    )

    errors = schema_errors(get_validator(REPORT_SCHEMA_PATH), report)
    if errors:
        print("ERROR: the report does not match the schema:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
