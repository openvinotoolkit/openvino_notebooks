"""Computes the `summary` block of nightly-report.json (see $defs/summary in schema/nightly-report.schema.json).

Stdlib-only. All functions are pure: the input report is never mutated.
"""

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Optional

from nightly_report.constants import NEAR_TIMEOUT_RATIO, Status

TOP_SLOWEST = 10
TOP_ERROR_GROUPS = 10
UNKNOWN_VERSION = "unknown"


def _rate(numerator: float, denominator: float) -> Optional[float]:
    if not denominator:
        return None
    # Clamp guards the schema bound [0, 1] against inconsistent input (e.g. results for notebooks missing from the catalog).
    return round(min(numerator / denominator, 1.0), 4)


def _seconds(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 1)


def _sum_seconds(values: Iterable[Optional[float]]) -> float:
    return round(float(sum(v for v in values if v is not None)), 1)


def _is_executed(result: dict) -> bool:
    return result.get("status") in Status.EXECUTED


def _sorted_counter(counter: Counter) -> dict:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))))


def _counts(results: list, jobs: list) -> dict:
    by_status = Counter(r.get("status") for r in results)
    executed = sum(by_status[s] for s in Status.EXECUTED)
    return {
        "planned": len(results),
        "executed": executed,
        "passed": by_status[Status.PASSED],
        "failed": by_status[Status.FAILED],
        "timeout": by_status[Status.TIMEOUT],
        "error": by_status[Status.ERROR],
        "skipped": by_status[Status.SKIPPED],
        "not_run": by_status[Status.NOT_RUN],
        "pass_rate": _rate(by_status[Status.PASSED], executed),
        "test_s": _sum_seconds(r.get("duration_s") for r in results if _is_executed(r)),
        "compute_s": _sum_seconds(j.get("duration_s") for j in jobs),
    }


def _grouped_counts(results: list, jobs: list, configs: dict, key: Callable[[dict], Any], keys: Iterable[str]) -> dict:
    """Counts per group, where `key(config)` maps a config to its group; results/jobs of unknown configs are ignored."""
    grouped_results = {k: [] for k in keys}
    grouped_jobs = {k: [] for k in keys}
    for result in results:
        config = configs.get(result.get("config_id"))
        if config is not None:
            grouped_results.setdefault(key(config), []).append(result)
    for job in jobs:
        config = configs.get(job.get("config_id"))
        if config is not None:
            grouped_jobs.setdefault(key(config), []).append(job)
    groups = list(grouped_results) + [k for k in grouped_jobs if k not in grouped_results]
    return {group: _counts(grouped_results.get(group, []), grouped_jobs.get(group, [])) for group in groups}


def _by_config(results: list, jobs: list, configs: dict) -> dict:
    results_by_id = defaultdict(list)
    jobs_by_id = defaultdict(list)
    for result in results:
        results_by_id[result.get("config_id")].append(result)
    for job in jobs:
        jobs_by_id[job.get("config_id")].append(job)
    # Every configured id is present (with zero counts if needed); ids referenced only by results/jobs are kept too.
    ids = list(configs)
    ids += sorted({str(i) for i in list(results_by_id) + list(jobs_by_id) if i is not None and i not in configs})
    return {config_id: _counts(results_by_id.get(config_id, []), jobs_by_id.get(config_id, [])) for config_id in ids}


def _platform_specific(executed_by_notebook: dict, configs: dict, attribute: str) -> dict:
    found = defaultdict(list)
    for notebook, results in executed_by_notebook.items():
        statuses = defaultdict(list)
        for result in results:
            config = configs.get(result.get("config_id"))
            if config is not None and config.get(attribute) is not None:
                statuses[config[attribute]].append(result["status"])
        failing = [value for value, values in statuses.items() if any(s in Status.FAILING for s in values)]
        passing = [value for value, values in statuses.items() if all(s == Status.PASSED for s in values)]
        if len(failing) == 1 and passing:
            found[failing[0]].append(notebook)
    return {value: sorted(notebooks) for value, notebooks in sorted(found.items())}


def _duration_entry(result: dict) -> dict:
    duration = result["duration_s"]
    timeout = result.get("timeout_s")
    return {
        "notebook": result.get("notebook"),
        "config_id": result.get("config_id"),
        "duration_s": _seconds(duration),
        "timeout_s": _seconds(timeout),
        "ratio": round(duration / timeout, 3) if timeout else None,
    }


def _p90(values: list) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def _durations(run: dict, jobs: list, executed: list) -> dict:
    compute_s = _sum_seconds(j.get("duration_s") for j in jobs)
    test_s = _sum_seconds(r.get("duration_s") for r in executed)
    timed = [r for r in executed if r.get("duration_s") is not None]
    passed = [r for r in timed if r["status"] == Status.PASSED]
    passed_durations = [r["duration_s"] for r in passed]

    slowest = sorted(timed, key=lambda r: (-r["duration_s"], r.get("notebook") or "", r.get("config_id") or ""))[:TOP_SLOWEST]
    near_timeout = [_duration_entry(r) for r in passed if r.get("timeout_s") and r["duration_s"] >= NEAR_TIMEOUT_RATIO * r["timeout_s"]]
    near_timeout.sort(key=lambda e: (-e["ratio"], e["notebook"] or "", e["config_id"] or ""))

    return {
        "run_wall_clock_s": _seconds(run.get("wall_clock_s")),
        "compute_s": compute_s,
        "test_s": test_s,
        "setup_overhead_s": round(max(compute_s - test_s, 0.0), 1),
        "notebook_median_s": _seconds(statistics.median(passed_durations)) if passed_durations else None,
        "notebook_p90_s": _seconds(_p90(passed_durations)),
        "notebook_max_s": _seconds(max(passed_durations)) if passed_durations else None,
        "slowest": [_duration_entry(r) for r in slowest],
        "near_timeout": near_timeout,
    }


def _notebooks(catalog: list, executed_by_notebook: dict) -> dict:
    tested = set(executed_by_notebook)
    passing_everywhere = [n for n, rs in executed_by_notebook.items() if all(r["status"] == Status.PASSED for r in rs)]
    failing_somewhere = sorted(n for n, rs in executed_by_notebook.items() if any(r["status"] in Status.FAILING for r in rs))
    failing_everywhere = [n for n, rs in executed_by_notebook.items() if not any(r["status"] == Status.PASSED for r in rs)]
    untested_list = sorted({nb.get("path") for nb in catalog if nb.get("path") is not None} - tested)
    return {
        "total": len(catalog),
        "tested": len(tested),
        # Equals total - tested when results only reference catalog notebooks; never negative otherwise.
        "untested": len(untested_list),
        "untested_list": untested_list,
        "passing_everywhere": len(passing_everywhere),
        "failing_somewhere": len(failing_somewhere),
        "failing_everywhere": len(failing_everywhere),
        "failing_list": failing_somewhere,
    }


def _failures(failures: list, error_groups: list) -> dict:
    return {
        "by_category": _sorted_counter(Counter(f.get("category") or "unknown" for f in failures)),
        "by_error_class": _sorted_counter(Counter(f.get("error_class") or "other" for f in failures)),
        "top_error_groups": [
            {
                "signature": group.get("signature"),
                "occurrences": group.get("occurrences"),
                "exception_type": group.get("exception_type"),
                "exception_message": group.get("exception_message"),
            }
            for group in error_groups[:TOP_ERROR_GROUPS]
        ],
    }


def _openvino_version(result: dict) -> Optional[str]:
    return (result.get("packages") or {}).get("openvino") or None


def _infra(jobs: list, results: list) -> dict:
    conclusions = Counter(j.get("conclusion") for j in jobs)
    failed_before_tests = sum(1 for j in jobs if not j.get("tests_started", True))
    return {
        "jobs_total": len(jobs),
        "jobs_succeeded": conclusions["success"],
        "jobs_failed": conclusions["failure"] + conclusions["timed_out"],
        "jobs_cancelled": conclusions["cancelled"],
        "jobs_failed_before_tests": failed_before_tests,
        "notebooks_lost": sum(1 for r in results if r.get("status") == Status.NOT_RUN),
        "infra_failure_rate": _rate(failed_before_tests, len(jobs)),
    }


def compute_summary(report: dict) -> dict:
    """Returns the `summary` block for a report holding run, configs, jobs, notebooks, results, failures and error_groups."""
    run = report.get("run") or {}
    config_list = report.get("configs") or []
    jobs = report.get("jobs") or []
    catalog = report.get("notebooks") or []
    results = report.get("results") or []
    failures = report.get("failures") or []
    error_groups = report.get("error_groups") or []

    configs = {c["id"]: c for c in config_list}
    executed = [r for r in results if _is_executed(r)]
    executed_by_notebook = defaultdict(list)
    for result in executed:
        executed_by_notebook[result.get("notebook")].append(result)

    totals = _counts(results, jobs)
    notebooks = _notebooks(catalog, executed_by_notebook)
    versions = [_openvino_version(r) for r in executed]
    known_versions = [v for v in versions if v is not None]

    return {
        "notebooks": notebooks,
        "results": totals,
        "pass_rate": totals["pass_rate"],
        "notebook_pass_rate": _rate(notebooks["passing_everywhere"], notebooks["tested"]),
        "coverage": _rate(totals["executed"], totals["planned"] - totals["skipped"]),
        "matrix_coverage": _rate(totals["executed"], len(catalog) * len(configs)),
        "platform_specific_failures": {
            "os": _platform_specific(executed_by_notebook, configs, "os"),
            "python": _platform_specific(executed_by_notebook, configs, "python"),
        },
        "by_config": _by_config(results, jobs, configs),
        "by_os": _grouped_counts(results, jobs, configs, lambda c: c.get("os"), dict.fromkeys(c.get("os") for c in config_list)),
        "by_python": _grouped_counts(results, jobs, configs, lambda c: c.get("python"), dict.fromkeys(c.get("python") for c in config_list)),
        "infra": _infra(jobs, results),
        "durations": _durations(run, jobs, executed),
        "failures": _failures(failures, error_groups),
        "openvino_versions": _sorted_counter(Counter(v or UNKNOWN_VERSION for v in versions)),
        "openvino_dev_share": _rate(sum(1 for v in known_versions if "dev" in v.lower()), len(known_versions)),
    }
