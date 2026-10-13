"""Renders a nightly report as GitHub-flavoured markdown (for $GITHUB_STEP_SUMMARY and summary.md).

Stdlib-only. Tolerates missing optional fields: anything absent or null is rendered as "n/a" or omitted.
"""

import html
import re
from typing import Iterable, Optional

from nightly_report.constants import NEAR_TIMEOUT_RATIO
from nightly_report.metrics import compute_summary

MAX_FAILURE_ROWS = 50
MAX_ERROR_GROUPS = 10
MAX_GROUP_NOTEBOOKS = 5
ERROR_MAX_CHARS = 120
MAX_LOG_EXCERPTS = 20
LOG_EXCERPT_MAX_LINES = 40
LOG_EXCERPT_LINE_MAX_CHARS = 300

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


def _legend(items: list) -> list:
    """Explanation of the columns of the table above it: (column names, description) pairs."""
    return [f"- **{columns}**: {description}" for columns, description in items] + [""]


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


def _headline(report: dict, summary: dict) -> list:
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
    timeout_s = ((report.get("run") or {}).get("inputs") or {}).get("timeout_s")
    timeout = f" ({format_duration(timeout_s)})" if timeout_s else ""
    legend = [
        (
            "Pass rate",
            "passed ÷ executed, counting every notebook × configuration result. "
            "Executed = passed + failed + timeout + error; skipped and not run results are excluded.",
        ),
        ("Notebook pass rate", "notebooks that passed on every configuration they ran on ÷ notebooks that ran on at least one configuration."),
        (
            "Coverage",
            "executed ÷ (planned − skipped): the share of notebook runs that should have happened and did. "
            "It drops when test jobs fail before the tests start or are cancelled.",
        ),
        (
            "Executed, Passed, Failed, Timeout, Error, Not run",
            "number of notebook × configuration results with that status. "
            "_Failed_: the notebook failed (exception in a cell, crashed kernel, …). "
            f"_Timeout_: the notebook was killed after the per-notebook timeout{timeout}. "
            "_Error_: a problem of the test harness (e.g. creating the notebook environment), not of the notebook. "
            "_Not run_: planned, but never executed because its test job failed before the tests, was cancelled or was interrupted.",
        ),
        ("Untested notebooks", "notebooks of the repository without any executed result in this run (skipped or not run everywhere, or not selected)."),
        ("Wall clock", "time from the start of the run to the end of the last test job."),
    ]
    return _table(headers, [row]) + [""] + _legend(legend)


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
    legend = [
        ("Config", "test configuration `<device>-<os>-py<python>`."),
        ("Executed, Passed, Failed, Timeout, Error, Not run", "as in the table above, for this configuration only."),
        ("Skipped", "notebooks excluded on purpose for this configuration by `.ci/skipped_notebooks.yml` or an ignore list (e.g. `.ci/tensorflow.txt`)."),
        ("Pass rate", "passed ÷ executed for this configuration."),
        ("Test time", "sum of the execution times of the executed notebooks of this configuration (without the setup of the test jobs)."),
    ]
    return lines + _table(headers, rows) + [""] + _legend(legend)


def _failure_links(failure: dict) -> str:
    links = failure.get("links") or {}
    labeled = (
        ("failure log", links.get("log_excerpt")),
        ("job", links.get("job")),
        ("logs", links.get("logs_artifact")),
        ("source", links.get("notebook_source")),
    )
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
    legend = [
        ("Status", "`failed`, `timeout` or `error` (see the first table)."),
        (
            "Category / class",
            "what the notebook output shows (`cell_error`: exception in a cell, `kernel_died`, `timeout`, `harness_error`, `unknown`) / "
            "heuristic guess of the cause (`dependency`, `network`, `memory`, `disk`, `openvino`, `assertion`, `crash`, `timeout`, `harness`, `other`).",
        ),
        ("Error", f"exception type and message parsed from the notebook output (first {ERROR_MAX_CHARS} characters)."),
        (
            "Links",
            "_failure log_: the job log opened at the failure of the notebook (the piece is shown under *Failure logs*); "
            "_job_: log of the test job; _logs_: artifact with the full output of the notebooks that did not pass; "
            "_source_: the notebook at the tested commit.",
        ),
    ]
    return lines + [""] + _legend(legend)


def _fence(text: str) -> str:
    """A code fence longer than any tilde run at the start of a line of the text."""
    longest = max((len(match.group(0)) for match in re.finditer(r"(?m)^~+", text)), default=0)
    return "~" * max(3, longest + 1)


def _failure_logs(report: dict) -> list:
    failures = sorted(report.get("failures") or [], key=lambda f: (f.get("notebook") or "", f.get("config_id") or ""))
    failures = [f for f in failures if f.get("log_excerpt")]
    if not failures:
        return []
    lines = ["## Failure logs", "", "The piece of the notebook output around each failure; _open in the job log_ shows it in the full log of the job.", ""]
    for failure in failures[:MAX_LOG_EXCERPTS]:
        excerpt_lines = str(failure["log_excerpt"]).splitlines()
        shown = [_truncate(line, LOG_EXCERPT_LINE_MAX_CHARS) for line in excerpt_lines[:LOG_EXCERPT_MAX_LINES]]
        if len(excerpt_lines) > LOG_EXCERPT_MAX_LINES:
            shown.append(f"… {len(excerpt_lines) - LOG_EXCERPT_MAX_LINES} more lines in the job log")
        text = "\n".join(shown)
        numbers = failure.get("log_excerpt_lines") or {}
        where = f" (lines {numbers['start']}–{numbers['end']})" if numbers.get("start") and numbers.get("end") else ""
        summary = " · ".join(html.escape(str(part)) for part in (failure.get("notebook"), failure.get("config_id"), failure.get("status")) if part)
        url = (failure.get("links") or {}).get("log_excerpt")
        fence = _fence(text)
        lines += [f"<details><summary>{summary}{where}</summary>", ""]
        if url:
            lines += [_link("open in the job log", url), ""]
        lines += [fence + "text", text, fence, "", "</details>", ""]
    if len(failures) > MAX_LOG_EXCERPTS:
        lines += [f"…and {len(failures) - MAX_LOG_EXCERPTS} more in nightly-report.json", ""]
    return lines


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
    legend = [
        (
            "Signature",
            "hash of the failure category, exception type and message with paths, numbers and hashes masked. "
            "Failures with the same signature most likely share a cause, also across nights.",
        ),
        ("Occurrences", "number of failed notebook × configuration results with this signature."),
        ("Error", "exception type and message of the first affected notebook."),
        ("Notebooks", f"affected notebooks (first {MAX_GROUP_NOTEBOOKS})."),
    ]
    return ["## Error groups", ""] + _table(["Signature", "Occurrences", "Error", "Notebooks"], rows) + [""] + _legend(legend)


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
    intro = "Test jobs that stopped before running notebooks and the number of notebook × configuration results that were planned but not run."
    return ["## Infrastructure", "", intro, ""] + items + [""] if items else []


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
    intro = f"Passed notebooks that used at least {NEAR_TIMEOUT_RATIO:.0%} of the timeout and may start timing out."
    legend = [
        ("Duration", "execution time of the notebook."),
        ("Timeout", "per-notebook timeout."),
        ("Of timeout", "duration ÷ timeout."),
    ]
    return ["## Near timeout", "", intro, ""] + _table(["Notebook", "Config", "Duration", "Timeout", "Of timeout"], rows) + [""] + _legend(legend)


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
    lines += _headline(report, summary)
    lines += _by_config(summary)
    lines += _failures(report)
    lines += _failure_logs(report)
    lines += _error_groups(report)
    lines += _infrastructure(report, summary)
    lines += _near_timeout(summary)
    lines += _platform_specific(summary)
    return "\n".join(lines).rstrip("\n") + "\n"
