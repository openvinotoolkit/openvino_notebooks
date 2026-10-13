"""Locates the output of a failed notebook in a GitHub Actions job log and extracts the piece around the failure.

Line numbers follow the GitHub log viewer, so `<job url>#step:<step>:<line>` opens the piece: lines are numbered from 1
starting at the `##[group]Run ...` header of the step and counting every raw line (including `##[endgroup]`).
Raw lines are separated by "\n" only (Windows logs contain CRLF and lone CR characters inside lines). Stdlib only.
"""

import re
from typing import Optional

from nightly_report.error_parser import strip_ansi

STEP_HEADER = "##[group]Run "
NOTEBOOK_START = "Testing notebook: "
TREON_START = "Triggered test for "
# Lines after the output of the last notebook of a step: workflow commands and the final summary of validate_notebooks.py.
SECTION_END = (NOTEBOOK_START, "##[", "FAILED: ", "FAILED BY TIMEOUT: ", "Test report built with ")
# Lines printed by validate_notebooks.py after the notebook process exits (the end of the piece when treon prints no FAILED line).
RUN_END = ("OpenVINO after notebook execution", "TEST DURATION [")
TREON_SEPARATOR = "-----"

CONTEXT_BEFORE = 5
EXCERPT_MAX_LINES = 120
EXCERPT_MAX_CHARS = 12 * 1024

_TIMESTAMP_RE = re.compile(r"^\ufeff?(?:\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?)?")
# First line of the failure in the output of validate_notebooks.py / treon, see error_parser for the formats.
_FAILURE_RE = re.compile(
    r"ERROR in testing |Kernel died|Traceback \(most recent call last\)|timeout reached \(|Error during testing notebook"
    r"|Failed to create virtual environment|Error running Notebook test|\s-- FAILED",
    re.IGNORECASE,
)


def split_log(text: str) -> list[str]:
    """Lines of a job log without the timestamp prefixes, numbered like the GitHub log viewer (index 0 = raw line 1)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [_TIMESTAMP_RE.sub("", line.rstrip("\r"), count=1) for line in lines]


def _notebook_start(lines: list[str], notebook: str) -> Optional[int]:
    """Index of the 'Testing notebook: <path>' line of the notebook (path relative to the notebooks directory)."""
    exact, suffix = None, None
    for index, line in enumerate(lines):
        if not line.startswith(NOTEBOOK_START):
            continue
        path = line[len(NOTEBOOK_START) :].strip().replace("\\", "/")
        if exact is None and path.endswith("/notebooks/" + notebook):
            exact = index
        elif suffix is None and (path == notebook or path.endswith("/" + notebook)):
            suffix = index
    return exact if exact is not None else suffix


def _treon_name(line: str, basename: str) -> Optional[str]:
    """Name of the notebook in a 'Triggered test for <name>' line when it is the (test_-prefixed copy of the) notebook."""
    if not line.startswith(TREON_START):
        return None
    name = line[len(TREON_START) :].strip()
    file_name = name.replace("\\", "/").rsplit("/", 1)[-1]
    return name if file_name in (basename, "test_" + basename) else None


def _treon_start(lines: list[str], start: int, end: int, basename: str) -> tuple[Optional[int], Optional[str]]:
    """(index, name) of the last 'Triggered test for <notebook>' line in lines[start:end]."""
    for index in range(end - 1, start - 1, -1):
        name = _treon_name(lines[index], basename)
        if name:
            return index, name
    return None, None


def _span(lines: list[str], notebook: str) -> Optional[tuple[int, int, int, Optional[int]]]:
    """(step header, first, end (exclusive), FAILED line or None) of the output of the failed notebook.

    The piece starts at 'Triggered test for <notebook>' and ends at '<notebook>     -- FAILED' followed by the output treon
    prints with it (up to its '-----' result table). Without a FAILED line (cell error, dead kernel, timeout), it ends where
    the notebook process exits. Without a 'Triggered test for' line (the notebook did not start), it is the notebook section.
    """
    basename = notebook.replace("\\", "/").rsplit("/", 1)[-1]
    section = _notebook_start(lines, notebook)
    if section is not None:
        section_end = next((i for i in range(section + 1, len(lines)) if lines[i].startswith(SECTION_END)), len(lines))
        first, name = _treon_start(lines, section, section_end, basename)
        first = section if first is None else first
    else:  # a log without validate_notebooks.py sections: only treon output
        first, name = _treon_start(lines, 0, len(lines), basename)
        if first is None:
            return None
        section_end = next(
            (i for i in range(first + 1, len(lines)) if lines[i].startswith(SECTION_END + (TREON_START,))), len(lines)
        )
    step = next((i for i in range(first, -1, -1) if lines[i].startswith(STEP_HEADER)), None)
    if step is None:
        return None
    failed = None
    if name:
        failed_re = re.compile(rf"^\s*{re.escape(name)}\s+-- FAILED\b")
        failed = next((i for i in range(first + 1, section_end) if failed_re.match(lines[i])), None)
    if failed is not None:
        end = next((i for i in range(failed + 1, section_end) if lines[i].lstrip().startswith(TREON_SEPARATOR)), section_end)
    else:
        end = next((i for i in range(first + 1, section_end) if lines[i].startswith(RUN_END)), section_end)
    while end > first + 1 and not lines[end - 1].strip():
        end -= 1
    return step, first, end, failed


def _excerpt(lines: list[str], start: int, end: int, anchor: Optional[int] = None) -> tuple[int, int, Optional[int], str]:
    """(first, last, failure line or None, text) of the piece of lines[start:end] (indexes).

    The whole span when it is short enough, otherwise EXCERPT_MAX_LINES lines ending at the end of the span, moved up to
    start CONTEXT_BEFORE lines before the failure line when needed.
    """
    if anchor is None:
        anchor = next((index for index in range(start, end) if _FAILURE_RE.search(lines[index])), None)
    first = max(start, end - EXCERPT_MAX_LINES)
    if anchor is not None:
        first = max(start, min(first, anchor - CONTEXT_BEFORE))
    last = min(end, first + EXCERPT_MAX_LINES) - 1
    text = "\n".join(strip_ansi(line).rstrip() for line in lines[first : last + 1]).strip("\n")
    if len(text) > EXCERPT_MAX_CHARS:
        text = text[:EXCERPT_MAX_CHARS].rstrip() + "\n…"
    return first, last, anchor, text


def failure_excerpt(log_text: str, notebook: str) -> Optional[dict]:
    """The failure piece of `notebook` in a job log, or None when the notebook output is not found.

    Returns {"text", "start", "end", "error_line"} with line numbers of the GitHub log viewer, relative to the step that
    ran the notebook (error_line is None when no failure marker was found).
    """
    lines = split_log(log_text or "")
    span = _span(lines, notebook)
    if span is None:
        return None
    step, start, end, failed = span
    first, last, anchor, text = _excerpt(lines, start, end, failed)
    if not text:
        return None
    return {
        "text": text,
        "start": first - step + 1,
        "end": last - step + 1,
        "error_line": anchor - step + 1 if anchor is not None else None,
    }


def file_excerpt(log_text: str) -> Optional[str]:
    """The failure piece of a captured notebook log (the whole file is the notebook output, from 'Triggered test for' if any)."""
    lines = split_log(log_text or "")
    start = next((index for index, line in enumerate(lines) if line.startswith(TREON_START)), 0)
    if start >= len(lines):
        return None
    return _excerpt(lines, start, len(lines))[3] or None
