"""Representative notebook logs, infrastructure logs, and environment evidence."""

from __future__ import annotations

import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from collector_common import (
    ANSI_RE,
    ARTIFACT_ID_RE,
    CORE_PACKAGES,
    JOB_LOG_TIMESTAMP_RE,
    MAX_JOB_LOG_BYTES,
    MAX_LOG_BYTES,
    MAX_TOTAL_LOG_BYTES,
    Context,
    Group,
    Member,
    CollectorError,
    Key,
)
from collector_github import download_artifact, gh, read_zip_member


def make_context(repo: str, source_repo: str, report: dict, environments: dict) -> Context:
    results = [r for r in report.get("results") or [] if r.get("notebook") and r.get("config_id")]
    by_notebook: Dict[str, List[dict]] = defaultdict(list)
    for result in results:
        by_notebook[result["notebook"]].append(result)
    return Context(
        repo=repo,
        source_repo=source_repo,
        run=report.get("run") or {},
        report=report,
        environments=environments,
        configs={c["id"]: c for c in report.get("configs") or [] if "id" in c},
        jobs={j["id"]: j for j in report.get("jobs") or [] if isinstance(j.get("id"), int)},
        results_by_key={(r["notebook"], r["config_id"]): r for r in results},
        results_by_notebook=dict(by_notebook),
    )


def config_preference(config_id: str) -> int:
    return 0 if config_id.startswith("cpu-ubuntu") else 1 if config_id.startswith("cpu") else 2


def pick_representatives(group: Group, limit: int) -> List[Member]:
    """Distinct notebooks first (Linux CPU preferred, widest impact first), then distinct configs."""
    ordered = sorted(group.members, key=lambda m: (config_preference(m.config_id), m.config_id, m.notebook))
    by_notebook: Dict[str, List[Member]] = defaultdict(list)
    for member in ordered:
        by_notebook[member.notebook].append(member)
    notebooks = sorted(by_notebook, key=lambda n: (-len(by_notebook[n]), n))
    picked = [by_notebook[n][0] for n in notebooks[:limit]]
    seen_configs = {m.config_id for m in picked}
    for member in ordered:
        if len(picked) >= limit:
            break
        if member not in picked and member.config_id not in seen_configs:
            picked.append(member)
            seen_configs.add(member.config_id)
    return picked


def pick_infra_jobs(group: Group, ctx: Context, limit: int) -> List[dict]:
    """Distinct failed jobs of an infra group, distinct configs first."""
    counts = Counter(m.job_id for m in group.members if m.job_id in ctx.jobs)
    jobs = sorted((ctx.jobs[job_id] for job_id in counts), key=lambda j: (config_preference(j.get("config_id", "")), j.get("config_id", ""), j["id"]))
    picked: List[dict] = []
    seen_configs: set = set()
    for job in jobs:
        if job.get("config_id") not in seen_configs:
            picked.append(job)
            seen_configs.add(job.get("config_id"))
    picked.extend(job for job in jobs if job not in picked)
    return picked[:limit]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def normalize_log(data: bytes) -> str:
    return strip_ansi(data.decode("utf-8", errors="replace")).replace("\r\n", "\n").replace("\r", "\n")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")[:120] or "unnamed"


def parse_artifact_id(url: Optional[str]) -> Optional[int]:
    match = ARTIFACT_ID_RE.search(url or "")
    return int(match.group(1)) if match else None


def collect_raw_logs(ctx: Context, members: List[Member], logs_dir: Path, workdir: Path, used_bytes: List[int], warn: Callable[[str], None]) -> Dict[Key, str]:
    """Saves the raw treon log of each member from its fragment artifact; returns paths relative to the group directory."""
    saved: Dict[Key, str] = {}
    zips: Dict[int, Optional[Path]] = {}

    def fragment_zip(artifact_id: int) -> Optional[Path]:
        if artifact_id not in zips:
            path = workdir / f"fragment-{artifact_id}.zip"
            try:
                download_artifact(ctx.source_repo, artifact_id, path)
                zips[artifact_id] = path
            except (CollectorError, OSError) as error:
                warn(f"fragment artifact {artifact_id} unavailable: {error}")
                zips[artifact_id] = None
        return zips[artifact_id]

    for member in members:
        links = (member.failure or {}).get("links") or {}
        artifact_id, log_name = parse_artifact_id(links.get("logs_artifact")), links.get("log_file")
        archive = fragment_zip(artifact_id) if artifact_id and log_name else None
        if archive is None:
            continue
        try:
            data = read_zip_member(archive, log_name, MAX_LOG_BYTES)
        except (CollectorError, zipfile.BadZipFile, OSError) as error:
            warn(f"raw log of {member.notebook} on {member.config_id} unavailable: {error}")
            continue
        if data is None:
            continue
        if used_bytes[0] + len(data) > MAX_TOTAL_LOG_BYTES:
            warn("raw log size budget reached; remaining logs are not downloaded")
            break
        used_bytes[0] += len(data)
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / f"{safe_name(member.notebook)}__{safe_name(member.config_id)}.log"
        path.write_text(normalize_log(data), encoding="utf-8")
        saved[member.key] = f"logs/{path.name}"
    for zip_path in zips.values():
        if zip_path is not None:
            zip_path.unlink(missing_ok=True)
    return saved


def job_log_digest(text: str, tail: int = 80, context_before: int = 30, context_after: int = 5, max_blocks: int = 4) -> str:
    """Blocks around the first `##[error]` lines plus the end of a job log, without timestamps and ANSI codes."""
    lines = [JOB_LOG_TIMESTAMP_RE.sub("", strip_ansi(line)) for line in text.replace("\r\n", "\n").split("\n")]
    error_lines = [i for i, line in enumerate(lines) if "##[error]" in line][:max_blocks]
    parts: List[str] = []
    covered = -1
    for index in error_lines:
        start = max(index - context_before, covered + 1)
        end = min(index + context_after + 1, len(lines))
        if start < end:
            parts.append(f"--- lines {start + 1}-{end} of the job log ---\n" + "\n".join(lines[start:end]))
            covered = end - 1
    start = max(len(lines) - tail, covered + 1)
    if start < len(lines):
        parts.append(f"--- lines {start + 1}-{len(lines)} of the job log ---\n" + "\n".join(lines[start:]))
    return "\n\n".join(parts)


def collect_job_logs(ctx: Context, jobs: List[dict], logs_dir: Path, warn: Callable[[str], None]) -> Dict[int, str]:
    saved: Dict[int, str] = {}
    for job in jobs:
        try:
            text = gh(["api", f"repos/{ctx.source_repo}/actions/jobs/{job['id']}/logs"]).decode("utf-8", errors="replace")[:MAX_JOB_LOG_BYTES]
        except CollectorError as error:
            warn(f"job log of {job['id']} unavailable: {error}")
            continue
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / f"job-{job['id']}-extract.log"
        path.write_text(job_log_digest(text), encoding="utf-8")
        saved[job["id"]] = f"logs/{path.name}"
    return saved


def normalize_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def relevant_packages(env: dict, text: str, limit: int = 40) -> Dict[str, str]:
    """Packages of an environment that are named in the error text, plus the OpenVINO family and other core libraries."""
    tokens = {normalize_package(token) for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)}
    picked = {}
    for name, version in env.items():
        normalized = normalize_package(name)
        if normalized in CORE_PACKAGES or normalized.startswith("openvino") or normalized in tokens:
            picked[name] = version
    return dict(sorted(picked.items())[:limit])


def environment_diff(failing: dict, passing: dict, relevant: set, limit: int = 30) -> Tuple[List[Tuple[str, str, str]], int]:
    diffs = [(name, failing.get(name, "-"), passing.get(name, "-")) for name in sorted(set(failing) | set(passing)) if failing.get(name) != passing.get(name)]
    diffs.sort(key=lambda d: (normalize_package(d[0]) not in relevant, d[0]))
    return diffs[:limit], len(diffs)


def passing_reference(ctx: Context, member: Member) -> Optional[dict]:
    """A passing result of the same notebook, preferably on a similar config, whose environment is known."""
    candidates = [r for r in ctx.results_by_notebook.get(member.notebook, []) if r.get("status") == "passed" and r.get("env_id") in ctx.environments]
    if not candidates:
        return None
    wanted = ctx.configs.get(member.config_id) or {}

    def distance(result: dict) -> tuple:
        config = ctx.configs.get(result["config_id"]) or {}
        return (
            config.get("os_family") != wanted.get("os_family"),
            config.get("python") != wanted.get("python"),
            config.get("device") != wanted.get("device"),
            result["config_id"],
        )

    return min(candidates, key=distance)


def version_rows(ctx: Context, group: Group) -> List[dict]:
    """OpenVINO versions of the group's notebooks: failing with this error vs passing."""
    failing, notebooks = {m.key for m in group.members}, set(group.notebooks)
    counts: Dict[str, Counter] = defaultdict(Counter)
    for notebook in notebooks:
        for result in ctx.results_by_notebook.get(notebook, []):
            version = (result.get("packages") or {}).get("openvino") or "unknown"
            key = (result["notebook"], result["config_id"])
            bucket = "failed_with_error" if key in failing else "passed" if result.get("status") == "passed" else "other"
            counts[version][bucket] += 1
    rows = [{"openvino": version, "failed_with_error": c["failed_with_error"], "passed": c["passed"], "other": c["other"]} for version, c in counts.items()]
    return sorted(rows, key=lambda r: (-r["failed_with_error"], r["openvino"]))
