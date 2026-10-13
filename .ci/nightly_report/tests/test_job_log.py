import pytest

from nightly_report.job_log import CONTEXT_BEFORE, EXCERPT_MAX_CHARS, EXCERPT_MAX_LINES, failure_excerpt, file_excerpt, split_log


def job_log(lines: list[str], newline: str = "\n", bom: bool = True) -> str:
    """Raw job log text: every line prefixed with a timestamp, like the GitHub job logs API returns it."""
    text = newline.join(f"2026-10-12T08:55:{index % 60:02d}.1234567Z {line}" for index, line in enumerate(lines)) + newline
    return ("\ufeff" if bom else "") + text


SETUP = ["Current runner version: '2.338.0'", "##[group]Run actions/checkout@v7", "with:", "##[endgroup]", "Syncing repository"]
STEP = ['##[group]Run IGNORE_ARGS=""', 'IGNORE_ARGS=""', "shell: bash --noprofile --norc -e -o pipefail {0}", "env:", "  OS_NAME: ubuntu-22.04", "##[endgroup]"]
PASSED = ["Testing notebook: /__w/r/r/notebooks/a/a.ipynb", "RUN a/a.ipynb", "Start executing cell 0", "TEST DURATION [a.ipynb]: 20.00 seconds"]
CELL_ERROR = [
    "Testing notebook: /__w/r/r/notebooks/b/b.ipynb",
    "RUN b/b.ipynb",
    "Executing treon version 0.1.4",
    "Start executing cell 0",
    "Start executing cell 1",
    "Start executing cell 2",
    "Start executing cell 3",
    "\x1b[0;31mERROR in testing test_b.ipynb\x1b[0m",
    "An error occurred while executing the following cell:",
    "ValueError: boom",
    "TEST DURATION [b.ipynb]: 61.00 seconds",
]
TIMEOUT = [
    "Testing notebook: /__w/r/r/notebooks/c/nested/c.ipynb",
    "Start executing cell 7",
    "Notebook test [test_c.ipynb] timeout reached (1200s), killing process...",
    "Killed process group PID 42",
    "TEST DURATION [c.ipynb]: 1200.01 seconds",
]
UNKNOWN = ["Testing notebook: /__w/r/r/notebooks/d/d.ipynb", *[f"output line {i}" for i in range(200)], "TEST DURATION [d.ipynb]: 5 seconds"]
AFTER = ["##[error]Process completed with exit code 1.", "##[group]Run actions/upload-artifact@v7", "with:", "##[endgroup]"]


def lines_of(*blocks):
    return [line for block in blocks for line in block]


def line_of(lines: list[str], text: str, step_header: str = STEP[0]) -> int:
    """Expected GitHub log viewer line number of `text`, relative to the step header."""
    return lines.index(text) - lines.index(step_header) + 1


def test_split_log_strips_timestamps_bom_and_keeps_numbering():
    text = "\ufeff2026-10-12T08:53:29.8408434Z first\r\n2026-10-12T08:53:29.8Z sec\rond\r\nno timestamp\n"
    assert split_log(text) == ["first", "sec\rond", "no timestamp"]
    assert split_log("") == []


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_cell_error_excerpt_and_line_numbers(newline):
    lines = lines_of(SETUP, STEP, PASSED, CELL_ERROR, AFTER)
    piece = failure_excerpt(job_log(lines, newline), "b/b.ipynb")
    error_line = line_of(lines, CELL_ERROR[7])
    assert piece["error_line"] == error_line
    assert piece["start"] == error_line - CONTEXT_BEFORE
    assert piece["end"] == line_of(lines, CELL_ERROR[-1])
    text = piece["text"].splitlines()
    assert text[0] == CELL_ERROR[2] and text[CONTEXT_BEFORE] == "ERROR in testing test_b.ipynb" and text[-1] == CELL_ERROR[-1]
    assert "\x1b" not in piece["text"] and "2026-10-12T" not in piece["text"]


def test_excerpt_starts_at_the_notebook_section():
    lines = lines_of(
        STEP, ["Testing notebook: /__w/r/r/notebooks/b/b.ipynb", "Kernel died while waiting for execute reply.", "Traceback (most recent call last):"]
    )
    piece = failure_excerpt(job_log(lines), "b/b.ipynb")
    assert piece["start"] == line_of(lines, lines[len(STEP)]) and piece["error_line"] == piece["start"] + 1


def test_timeout_excerpt_of_nested_notebook_on_windows_paths():
    windows = [line.replace("/__w/r/r/notebooks/c/nested/c.ipynb", "C:\\runner\\_work\\r\\r\\notebooks\\c\\nested\\c.ipynb") for line in TIMEOUT]
    lines = lines_of(SETUP, STEP, windows, AFTER)
    piece = failure_excerpt(job_log(lines, "\r\n"), "c/nested/c.ipynb")
    assert piece["error_line"] == line_of(lines, TIMEOUT[2])
    assert piece["start"] == line_of(lines, windows[0])
    assert "timeout reached (1200s)" in piece["text"]


def test_unknown_failure_shows_the_end_of_the_output():
    lines = lines_of(STEP, UNKNOWN, AFTER)
    piece = failure_excerpt(job_log(lines), "d/d.ipynb")
    assert piece["error_line"] is None
    assert piece["end"] == line_of(lines, UNKNOWN[-1]) and piece["end"] - piece["start"] + 1 == EXCERPT_MAX_LINES
    assert piece["text"].splitlines()[-1] == UNKNOWN[-1]


def test_section_ends_at_the_next_step():
    lines = lines_of(STEP, ["Testing notebook: /__w/r/r/notebooks/e/e.ipynb", "Start executing cell 0"], AFTER[1:], ["Traceback (most recent call last):"])
    piece = failure_excerpt(job_log(lines), "e/e.ipynb")
    assert piece["error_line"] is None and "Traceback" not in piece["text"]


def test_exact_notebook_path_is_preferred_over_suffix():
    other = ["Testing notebook: /__w/r/r/notebooks/x/b/b.ipynb", "ERROR in testing test_b.ipynb (other)"]
    lines = lines_of(STEP, other, CELL_ERROR)
    piece = failure_excerpt(job_log(lines), "b/b.ipynb")
    assert "(other)" not in piece["text"] and piece["error_line"] == line_of(lines, CELL_ERROR[7])


def test_not_found_and_without_step_header():
    assert failure_excerpt(job_log(lines_of(STEP, PASSED)), "b/b.ipynb") is None
    assert failure_excerpt(job_log(CELL_ERROR), "b/b.ipynb") is None
    assert failure_excerpt("", "b/b.ipynb") is None
    assert failure_excerpt(None, "b/b.ipynb") is None


def test_excerpt_is_capped():
    lines = lines_of(STEP, ["Testing notebook: /__w/r/r/notebooks/b/b.ipynb", "Traceback (most recent call last):", *["x" * 1000] * 50])
    piece = failure_excerpt(job_log(lines), "b/b.ipynb")
    assert len(piece["text"]) <= EXCERPT_MAX_CHARS + 2 and piece["text"].endswith("…")


def test_file_excerpt():
    assert file_excerpt("\n".join(CELL_ERROR[1:])).splitlines()[CONTEXT_BEFORE] == "ERROR in testing test_b.ipynb"
    assert file_excerpt("") is None and file_excerpt(None) is None
