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
# Lines after the output of the last notebook of a step: workflow commands and the final summary of validate_notebooks.py.
SECTION_END = (NOTEBOOK_START, "##[", "FAILED: ", "FAILED BY TIMEOUT: ", "Test report built with ")

CONTEXT_BEFORE = 5
EXCERPT_MAX_LINES = 80
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


def _excerpt(lines: list[str], start: int, end: int) -> tuple[int, int, Optional[int], str]:
    """(first, last, failure line or None, text) of the piece of lines[start:end] around the failure (indexes)."""
    anchor = next((index for index in range(start, end) if _FAILURE_RE.search(lines[index])), None)
    first = max(start, anchor - CONTEXT_BEFORE) if anchor is not None else max(start, end - EXCERPT_MAX_LINES)
    last = min(end, first + EXCERPT_MAX_LINES) - 1
    text = "\n".join(strip_ansi(line).rstrip() for line in lines[first : last + 1]).strip("\n")
    if len(text) > EXCERPT_MAX_CHARS:
        text = text[:EXCERPT_MAX_CHARS].rstrip() + "\n…"
    return first, last, anchor, text


def failure_excerpt(log_text: str, notebook: str) -> Optional[dict]:
    """The failure piece of `notebook` in a job log, or None when the notebook output is not found.

    Returns {"text", "start", "end", "error_line"} with line numbers of the GitHub log viewer, relative to the step that
    ran the notebook (error_line is None when no failure marker was found; the piece is then the end of the output).
    """
    lines = split_log(log_text or "")
    start = _notebook_start(lines, notebook)
    if start is None:
        return None
    step_start = next((index for index in range(start, -1, -1) if lines[index].startswith(STEP_HEADER)), None)
    if step_start is None:
        return None
    end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith(SECTION_END)), len(lines))
    first, last, anchor, text = _excerpt(lines, start, end)
    if not text:
        return None
    return {
        "text": text,
        "start": first - step_start + 1,
        "end": last - step_start + 1,
        "error_line": anchor - step_start + 1 if anchor is not None else None,
    }


def file_excerpt(log_text: str) -> Optional[str]:
    """The failure piece of a captured notebook log (the whole file is the notebook output)."""
    lines = split_log(log_text or "")
    if not lines:
        return None
    return _excerpt(lines, 0, len(lines))[3] or None
