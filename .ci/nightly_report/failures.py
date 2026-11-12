"""Failure records of failing results (log analysis, log excerpt, links) and their grouping by error signature."""

import hashlib
from pathlib import Path
from typing import Optional

from nightly_report.constants import Status
from nightly_report.error_parser import analyze_failure, compute_signature
from nightly_report.github_api import GitHubApi
from nightly_report.job_log import failure_excerpt, file_excerpt
from nightly_report.report_common import Links, Warnings, first_line


class FailureCollector:
    def __init__(
        self, links: Links, head_sha: str, notebooks_dir: str, warnings: Warnings, job_logs_dir: Optional[Path] = None, api: Optional[GitHubApi] = None
    ) -> None:
        """Job logs are read from `job_logs_dir` (<job id>.log files), else downloaded with `api`, else not used."""
        self.links = links
        self.head_sha = head_sha
        self.notebooks_dir = notebooks_dir
        self.warnings = warnings
        self.job_logs_dir = job_logs_dir
        self.api = api
        self._job_logs: dict[int, Optional[str]] = {}

    def _job_log(self, job: dict) -> Optional[str]:
        """Log of a test job (cached); None when unavailable."""
        cache = self._job_logs
        if job["id"] in cache:
            return cache[job["id"]]
        text = None
        if self.job_logs_dir is not None:
            path = self.job_logs_dir / f"{job['id']}.log"
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = None
        elif self.api is not None and job["id"]:
            text = self.api.get_job_log(job["id"])
        cache[job["id"]] = text
        return text

    def _log_excerpt(self, job: dict, notebook: str, fragment_log: Optional[str]) -> dict:
        """Piece of the failed notebook output and the link to it in the job log (fragment log as fallback)."""
        step = job.get("_treon_step_number")
        if step:
            job_log = self._job_log(job)
            piece = failure_excerpt(job_log, notebook) if job_log else None
            if piece:
                lines = {key: piece[key] for key in ("start", "end", "error_line", "test_start")}
                test_start = self.links.step(job["url"], step, piece["test_start"]) if piece["test_start"] else None
                return {
                    "text": piece["text"],
                    "source": "job_log",
                    "lines": lines,
                    "url": self.links.step(job["url"], step, piece["start"]),
                    "test_start_url": test_start,
                }
            if job_log:
                self.warnings.add(f"Output of {notebook} not found in the log of job {job['id']}.")
        text = file_excerpt(fragment_log) if fragment_log else None
        return {"text": text, "source": "fragment_log" if text else None, "lines": None, "url": job["treon_step_url"], "test_start_url": None}

    def collect(self, failing: dict[str, tuple[dict, dict]], results: list[dict]) -> list[dict]:
        """Failure records of the failing results; `failing` maps failure_id -> (fragment result, job)."""
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
            try:
                excerpt = self._log_excerpt(job, row["notebook"], text)
            except Exception as error:  # the log excerpt must never break the report
                self.warnings.add(f"Log excerpt failed for {row['failure_id']}: {error}")
                excerpt = {"text": None, "source": None, "lines": None, "url": job["treon_step_url"], "test_start_url": None}
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
                    "log_excerpt": excerpt["text"],
                    "log_excerpt_source": excerpt["source"],
                    "log_excerpt_lines": excerpt["lines"],
                    "links": {
                        "job": job["url"],
                        "treon_step": job["treon_step_url"],
                        "log_excerpt": excerpt["url"],
                        "test_start": excerpt["test_start_url"],
                        "logs_artifact": job["logs_artifact_url"],
                        "log_file": log_file,
                        "notebook_source": self.links.blob(self.head_sha, f"{self.notebooks_dir}/{row['notebook']}"),
                    },
                }
            )
        return failures


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
