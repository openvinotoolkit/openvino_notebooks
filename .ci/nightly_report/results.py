"""Result rows of the test jobs: from their fragments, else from the batch plan with skip and not-run reasons."""

from pathlib import Path
from typing import Optional

from nightly_report.constants import NotRunReason, SkipReason, Status
from nightly_report.report_common import Warnings


class ResultCollector:
    def __init__(self, plan: Optional[dict], notebooks_dir: str, timeout_s: Optional[float], repo_root: Path, skip_config: str, warnings: Warnings) -> None:
        self.plan = plan
        self.notebooks_dir = notebooks_dir
        self.timeout_s = timeout_s
        self.repo_root = repo_root
        self.skip_config = skip_config
        self.warnings = warnings

    def _plan_batch(self, batch: Optional[str]) -> Optional[list[str]]:
        if not self.plan or not batch:
            return None
        notebooks = (self.plan.get("batches") or {}).get(batch)
        if notebooks is None:
            return None
        prefix = f"{self.plan.get('notebooks_dir', self.notebooks_dir).strip('/')}/"
        return [path[len(prefix) :] if path.startswith(prefix) else path for path in notebooks]

    def _skip_reasons(self, job: dict) -> dict[str, tuple[str, Optional[str]]]:
        """Notebook -> (skip reason, ignore list file name or None) for a job without a fragment."""
        try:
            from validate_notebooks import get_ignore_list_sources, resolve_skip_reasons
        except ImportError as error:
            self.warnings.add(f"Skip resolution is unavailable ({error}); skipped notebooks of crashed jobs are reported as not_run.")
            return {}
        validation_config = {"os": job["_os"], "python": job["_python"], "device": job["_test_device"]}
        ignore_lists = [str(self.repo_root / item) if item.endswith(".txt") else item for item in job["_ignore_lists"]]
        try:
            reasons = resolve_skip_reasons(validation_config, self.repo_root / self.skip_config, ignore_lists)
            sources = get_ignore_list_sources(ignore_lists, self.repo_root / self.notebooks_dir)
        except Exception as error:
            self.warnings.add(f"Skip resolution failed for {job['config_id']}: {error}")
            return {}
        prefix = f"{self.notebooks_dir}/"
        skips = {}
        for path, reason in reasons.items():
            notebook = path[len(prefix) :] if path.startswith(prefix) else path
            skips[notebook] = (reason, sources.get(notebook) if reason == SkipReason.IGNORE_LIST else None)
        return skips

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

    def collect(self, jobs: list[dict]) -> tuple[list[dict], dict[str, tuple[dict, dict]]]:
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
                    skip_reason, skip_source = skip_reasons[notebook]
                    add(self._row(base, notebook, Status.SKIPPED, timeout_s=self.timeout_s, skip_reason=skip_reason, skip_source=skip_source))
                else:
                    add(self._row(base, notebook, Status.NOT_RUN, timeout_s=self.timeout_s, not_run_reason=reason))
        results.sort(key=lambda row: (row["notebook"], row["config_id"]))
        return results, failing

    @staticmethod
    def _row(
        base: dict, notebook: str, status: str, timeout_s=None, result: Optional[dict] = None, skip_reason=None, skip_source=None, not_run_reason=None
    ) -> dict:
        result = result or {}
        duration = result.get("duration_s")
        return {
            "notebook": notebook,
            **base,
            "status": status,
            "skip_reason": result.get("skip_reason", skip_reason) if status == Status.SKIPPED else None,
            "skip_source": result.get("skip_source", skip_source) if status == Status.SKIPPED else None,
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
