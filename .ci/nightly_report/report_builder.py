"""Assembles nightly-report.json and environments.json from the plan, the fragments, the GitHub API data and the checkout."""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from nightly_report.constants import ENVS_DIR_NAME, FRAGMENT_ARTIFACT_PREFIX, SCHEMA_VERSION, config_id as make_config_id
from nightly_report.failures import FailureCollector, group_errors
from nightly_report.fragment import parse_pip_freeze
from nightly_report.github_api import GitHubApi
from nightly_report.metrics import compute_summary
from nightly_report.report_common import Links, Warnings, format_time, load_json, parse_time, seconds_between
from nightly_report.report_inputs import load_fragments
from nightly_report.repository import collect_notebooks, commit_entry, get_head_commit
from nightly_report.results import ResultCollector
from nightly_report.workflow import (
    TEST_JOB_PREFIX,
    job_ignore_lists,
    load_workflow,
    os_family,
    parse_job_name,
    runner_label_text,
    validation_device,
)

TREON_STEP_PREFIX = "Analysing with treon"


class ReportBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.warnings = Warnings()
        self.repo_root: Path = args.repo_root.resolve()
        self.notebooks_dir: str = args.notebooks_dir.strip("/")
        self.repository: str = args.repository
        self.run_id: int = args.run_id
        self._workflow: Optional[tuple[dict[str, Optional[dict]], set[str]]] = None
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
    def _load_workflow(self) -> tuple[dict[str, Optional[dict]], set[str]]:
        """Test job id -> workflow_job_spec with its expanded matrix and the jobs the test jobs depend on (read once)."""
        if self._workflow is None:
            self._workflow = load_workflow(self.repo_root, self.args.workflow_file, self.warnings)
        return self._workflow

    def _ignore_lists(self, job_key: Optional[str], device: str, os_name: str, python: str) -> list[str]:
        """Ignore lists passed to validate_notebooks.py by a test job: from its workflow definition, else --ignore_list of this script."""
        return job_ignore_lists(self._workflow_specs(), job_key, device, os_name, python, self.args.ignore_list)

    def _workflow_specs(self) -> dict[str, Optional[dict]]:
        return self._load_workflow()[0]

    def _workflow_matrices(self) -> dict[str, Optional[list[dict]]]:
        return {job_key: (spec["combinations"] if spec else None) for job_key, spec in self._workflow_specs().items()}

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
            matrix = parse_job_name(name, {key: spec for key, spec in self._workflow_specs().items() if spec})
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
        job_key = (api_job.get("name") or "").split(" ")[0] or None
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
            "_test_device": fragment_job.get("test_device") or validation_device(device),
            "_ignore_lists": self._ignore_lists(job_key, device, os_name, python),
            "_os": os_name,
            "_python": python,
            "_runner_label": runner_label_text(fragment_job.get("runner_label")) or (matrix or {}).get("runner_label"),
            "_container_image": fragment_job.get("container_image"),
            "_treon_step_number": treon_step.get("number") if treon_step else None,
            "_fragment": fragment,
        }

    # ---------------------------------------------------------------- results
    def collect_results(self, jobs: list[dict]) -> tuple[list[dict], dict[str, tuple[dict, dict]]]:
        """Returns result rows and failure_id -> (fragment result, job) for later failure analysis."""
        return ResultCollector(self.plan, self.notebooks_dir, self.timeout_s, self.repo_root, self.args.skip_config, self.warnings).collect(jobs)

    # ---------------------------------------------------------------- failures
    def collect_failures(self, failing: dict[str, tuple[dict, dict]], results: list[dict]) -> list[dict]:
        # With saved API responses (--jobs_json) job logs are not downloaded either.
        api = self.api if self.args.jobs_json is None else None
        collector = FailureCollector(self.links, self.head_sha, self.notebooks_dir, self.warnings, self.args.job_logs_dir, api)
        return collector.collect(failing, results)

    # ---------------------------------------------------------------- catalog
    def collect_notebooks(self, results: list[dict]) -> list[dict]:
        return collect_notebooks(self.repo_root, self.notebooks_dir, self.head_sha, self.args.latest_ref, self.links, results, self.warnings)

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
                # Union over the test jobs; per-config lists are in configs[].ignore_lists
                "ignore_lists": list(dict.fromkeys([*self.args.ignore_list, *(item for job in jobs for item in job["_ignore_lists"])])),
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
                    "test_device": job["_test_device"],
                    "os": job["_os"],
                    "os_family": os_family(job["_os"]),
                    "python": job["_python"],
                    "runner_label": job["_runner_label"],
                    "container_image": job["_container_image"],
                    "ignore_lists": list(job["_ignore_lists"]),
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
            "error_groups": group_errors(failures),
        }
        environments = self.build_environments(jobs, results)
        report["summary"] = compute_summary(report)
        return report, environments
