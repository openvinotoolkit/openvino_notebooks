"""Renders a nightly report as GitHub-flavoured markdown (for $GITHUB_STEP_SUMMARY and summary.md).

Stdlib-only. Tolerates missing optional fields: anything absent or null is rendered as "n/a" or omitted.
"""

import re
from typing import Iterable, Optional

from nightly_report.metrics import compute_summary

MAX_FAILURE_ROWS = 50
MAX_ERROR_GROUPS = 10
MAX_GROUP_NOTEBOOKS = 5
ERROR_MAX_CHARS = 120

_NEWLINES = re.compile(r"\s*[\r\n]+\s*")


def format_rate(rate: Optional[float]) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def format_duration(seconds: Optional[float]) -> str:
    """4h 53m / 12m 5s / 45s."""
    if seconds is None:
        return "n/a"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def single_line(value) -> str:
    return "" if value is None else _NEWLINES.sub(" ", str(value)).strip()


def escape_cell(value) -> str:
    return single_line(value).replace("|", "\\|")


def _plain(text: str) -> str:
    # GitHub drops unknown HTML tags, which would swallow e.g. "<module>" or "<class 'X'>" from error messages.
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _link(text: str, url: Optional[str]) -> str:
    return f"[{text}]({url})" if url else text


def _code(text) -> str:
    return f"`{text}`" if text not in (None, "") else ""


def _table(headers: list, rows: Iterable[list]) -> list:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(escape_cell(cell) for cell in row) + " |" for row in rows]
    return lines


def _error_text(exception_type: Optional[str], message: Optional[str]) -> str:
    parts = [single_line(p) for p in (exception_type, message) if p]
    return _plain(_truncate(": ".join(p for p in parts if p), ERROR_MAX_CHARS))


def _header(report: dict) -> list:
    run = report.get("run") or {}
    title = f"# Notebooks nightly — {run.get('nightly_date') or 'unknown date'}"
    if run.get("number") is not None:
        title += f" (run #{run['number']})"

    parts = []
    if run.get("url"):
        parts.append(_link("Run", run["url"]))
    commit = run.get("commit") or {}
    short_sha = commit.get("short_sha") or (commit.get("sha") or "")[:7]
    if short_sha:
        text = "commit " + _link(_code(short_sha), commit.get("url"))
        message = (commit.get("message") or "").strip().splitlines()
        if message:
            text += " " + _plain(message[0].strip())
        parts.append(text)
    if run.get("event"):
        parts.append(f"event {_code(run['event'])}")
    if run.get("branch"):
        parts.append(f"branch {_code(run['branch'])}")

    lines = [title, ""]
    if parts:
        lines += [" · ".join(parts), ""]
    return lines


def _headline(summary: dict) -> list:
    results = summary.get("results") or {}
    notebooks = summary.get("notebooks") or {}
    durations = summary.get("durations") or {}
    headers = ["Pass rate", "Notebook pass rate", "Coverage", "Executed", "Passed", "Failed", "Timeout", "Error", "Not run", "Untested notebooks", "Wall clock"]
    row = [
        format_rate(summary.get("pass_rate")),
        format_rate(summary.get("notebook_pass_rate")),
        format_rate(summary.get("coverage")),
        *(results.get(key, 0) for key in ("executed", "passed", "failed", "timeout", "error", "not_run")),
        notebooks.get("untested", 0),
        format_duration(durations.get("run_wall_clock_s")),
    ]
    return _table(headers, [row]) + [""]


def _by_config(summary: dict) -> list:
    lines = ["## By configuration", ""]
    by_config = summary.get("by_config") or {}
    if not by_config:
        return lines + ["_No configurations._", ""]
    headers = ["Config", "Executed", "Passed", "Failed", "Timeout", "Error", "Skipped", "Not run", "Pass rate", "Test time"]
    rows = [
        [
            config_id,
            *(counts.get(key, 0) for key in ("executed", "passed", "failed", "timeout", "error", "skipped", "not_run")),
            format_rate(counts.get("pass_rate")),
            format_duration(counts.get("test_s")),
        ]
        for config_id, counts in by_config.items()
    ]
    return lines + _table(headers, rows) + [""]


def _failure_links(failure: dict) -> str:
    links = failure.get("links") or {}
    labeled = (("job", links.get("job")), ("logs", links.get("logs_artifact")), ("source", links.get("notebook_source")))
    return " · ".join(_link(label, url) for label, url in labeled if url)


def _failures(report: dict) -> list:
    failures = sorted(report.get("failures") or [], key=lambda f: (f.get("notebook") or "", f.get("config_id") or ""))
    lines = [f"## Failures ({len(failures)})", ""]
    if not failures:
        return lines + ["_No failures._", ""]
    headers = ["Notebook", "Config", "Status", "Category / class", "Error", "Links"]
    rows = [
        [
            _code(f.get("notebook")),
            f.get("config_id") or "",
            f.get("status") or "",
            " / ".join(str(v) for v in (f.get("category"), f.get("error_class")) if v),
            _error_text(f.get("exception_type"), f.get("exception_message")),
            _failure_links(f),
        ]
        for f in failures[:MAX_FAILURE_ROWS]
    ]
    lines += _table(headers, rows)
    if len(failures) > MAX_FAILURE_ROWS:
        lines += ["", f"…and {len(failures) - MAX_FAILURE_ROWS} more"]
    return lines + [""]


def _error_groups(report: dict) -> list:
    groups = (report.get("error_groups") or [])[:MAX_ERROR_GROUPS]
    if not groups:
        return []
    rows = []
    for group in groups:
        notebooks = group.get("notebooks") or []
        shown = ", ".join(_code(n) for n in notebooks[:MAX_GROUP_NOTEBOOKS])
        if len(notebooks) > MAX_GROUP_NOTEBOOKS:
            shown += f" +{len(notebooks) - MAX_GROUP_NOTEBOOKS}"
        rows.append(
            [_code(group.get("signature")), group.get("occurrences", ""), _error_text(group.get("exception_type"), group.get("exception_message")), shown]
        )
    return ["## Error groups", ""] + _table(["Signature", "Occurrences", "Error", "Notebooks"], rows) + [""]


def _infrastructure(report: dict, summary: dict) -> list:
    items = []
    for job in report.get("jobs") or []:
        if job.get("tests_started", True):
            continue
        step = (job.get("failed_step") or {}).get("name") or "unknown"
        items.append(f"- {_link(single_line(job.get('name') or job.get('id') or 'job'), job.get('url'))} — failed step: {_plain(single_line(step))}")
    not_run = (summary.get("results") or {}).get("not_run", 0)
    if not_run:
        items.append(f"- Notebook runs lost (not run): {not_run}")
    return ["## Infrastructure", ""] + items + [""] if items else []


def _near_timeout(summary: dict) -> list:
    entries = (summary.get("durations") or {}).get("near_timeout") or []
    if not entries:
        return []
    rows = [
        [
            _code(e.get("notebook")),
            e.get("config_id") or "",
            format_duration(e.get("duration_s")),
            format_duration(e.get("timeout_s")),
            format_rate(e.get("ratio")),
        ]
        for e in entries
    ]
    return ["## Near timeout", ""] + _table(["Notebook", "Config", "Duration", "Timeout", "Of timeout"], rows) + [""]


def _platform_specific(summary: dict) -> list:
    specific = summary.get("platform_specific_failures") or {}
    items = []
    for key, label in (("os", "OS"), ("python", "Python")):
        for value, notebooks in (specific.get(key) or {}).items():
            items.append(f"- {label} {_code(value)}: " + ", ".join(_code(n) for n in notebooks))
    return (
        ["## Platform-specific failures", "", "Notebooks failing only on one OS or Python version while passing on others.", ""] + items + [""] if items else []
    )


def render_markdown(report: dict) -> str:
    """Markdown summary of a report; the summary block is computed when the report has none."""
    summary = report.get("summary") or compute_summary(report)
    lines = _header(report)
    lines += _headline(summary)
    lines += _by_config(summary)
    lines += _failures(report)
    lines += _error_groups(report)
    lines += _infrastructure(report, summary)
    lines += _near_timeout(summary)
    lines += _platform_specific(summary)
    return "\n".join(lines).rstrip("\n") + "\n"
