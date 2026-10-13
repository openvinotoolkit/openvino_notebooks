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
    "Triggered test for test_b.ipynb",
    "Start executing cell 0",
    "Start executing cell 1",
    "Start executing cell 2",
    "Start executing cell 3",
    "\x1b[0;31mERROR in testing test_b.ipynb\x1b[0m",
    "An error occurred while executing the following cell:",
    "ValueError: boom",
    "",
    "OpenVINO after notebook execution: 2026.4.1",
    "TEST DURATION [b.ipynb]: 61.00 seconds",
]
TREON_FAILED = [
    "Testing notebook: /__w/r/r/notebooks/f/f.ipynb",
    "Executing treon version 0.1.4",
    "-----------------------------------------------------------------------",
    "test_f.ipynb",
    "-----------------------------------------------------------------------",
    "",
    "Triggered test for test_f.ipynb",
    "Start executing cell 0",
    "Start executing cell 1",
    "",
    "test_f.ipynb     -- FAILED ",
    "AssertionError: unittest failed",
    "",
    "-----------------------------------------------------------------------",
    "TEST RESULT",
    "test_f.ipynb       -- FAILED ",
    "TEST DURATION [f.ipynb]: 10.00 seconds",
]
TIMEOUT = [
    "Testing notebook: /__w/r/r/notebooks/c/nested/c.ipynb",
    "Triggered test for test_c.ipynb",
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
def test_cell_error_piece_runs_from_trigger_to_process_exit(newline):
    lines = lines_of(SETUP, STEP, PASSED, CELL_ERROR, AFTER)
    piece = failure_excerpt(job_log(lines, newline), "b/b.ipynb")
    assert piece["start"] == line_of(lines, "Triggered test for test_b.ipynb")
    assert piece["error_line"] == line_of(lines, CELL_ERROR[8])
    assert piece["end"] == line_of(lines, "ValueError: boom")
    text = piece["text"].splitlines()
    assert text[0] == "Triggered test for test_b.ipynb" and text[-1] == "ValueError: boom"
    assert "ERROR in testing test_b.ipynb" in text and "TEST DURATION" not in piece["text"]
    assert "\x1b" not in piece["text"] and "2026-10-12T" not in piece["text"]


def test_treon_failed_piece_includes_the_failure_output():
    lines = lines_of(STEP, PASSED, TREON_FAILED, AFTER)
    piece = failure_excerpt(job_log(lines), "f/f.ipynb")
    assert piece["start"] == line_of(lines, "Triggered test for test_f.ipynb")
    assert piece["error_line"] == line_of(lines, "test_f.ipynb     -- FAILED ")
    assert piece["end"] == line_of(lines, "AssertionError: unittest failed")
    assert "TEST RESULT" not in piece["text"]


def test_dead_kernel_without_failed_line():
    lines = lines_of(
        STEP,
        ["Testing notebook: /__w/r/r/notebooks/b/b.ipynb", "Triggered test for test_b.ipynb", "Start executing cell 0"],
        ["Kernel died while waiting for execute reply.", "Traceback (most recent call last):", "DeadKernelError: Kernel died"],
        ["TEST DURATION [b.ipynb]: 5 seconds"],
    )
    piece = failure_excerpt(job_log(lines), "b/b.ipynb")
    assert piece["start"] == line_of(lines, "Triggered test for test_b.ipynb")
    assert piece["error_line"] == line_of(lines, "Kernel died while waiting for execute reply.")
    assert piece["end"] == line_of(lines, "DeadKernelError: Kernel died")


def test_notebook_that_did_not_start_shows_its_section():
    lines = lines_of(STEP, ["Testing notebook: /__w/r/r/notebooks/b/b.ipynb", "Failed to create virtual environment: boom"], AFTER)
    piece = failure_excerpt(job_log(lines), "b/b.ipynb")
    assert piece["start"] == line_of(lines, lines[len(STEP)]) and piece["error_line"] == piece["start"] + 1


def test_timeout_excerpt_of_nested_notebook_on_windows_paths():
    windows = [line.replace("/__w/r/r/notebooks/c/nested/c.ipynb", "C:\\runner\\_work\\r\\r\\notebooks\\c\\nested\\c.ipynb") for line in TIMEOUT]
    lines = lines_of(SETUP, STEP, windows, AFTER)
    piece = failure_excerpt(job_log(lines, "\r\n"), "c/nested/c.ipynb")
    assert piece["start"] == line_of(lines, "Triggered test for test_c.ipynb")
    assert piece["error_line"] == line_of(lines, TIMEOUT[3])
    assert "timeout reached (1200s)" in piece["text"] and "TEST DURATION" not in piece["text"]


def test_treon_output_without_harness_sections():
    lines = lines_of(STEP, TREON_FAILED[1:-1], ["Triggered test for test_g.ipynb", "test_g.ipynb     -- PASSED "], AFTER)
    piece = failure_excerpt(job_log(lines), "f/f.ipynb")
    assert piece["start"] == line_of(lines, "Triggered test for test_f.ipynb")
    assert piece["error_line"] == line_of(lines, "test_f.ipynb     -- FAILED ")


def test_long_output_keeps_the_end_and_the_failure_line():
    cells = [f"Start executing cell {i}" for i in range(300)]
    lines = lines_of(STEP, ["Testing notebook: /__w/r/r/notebooks/d/d.ipynb", "Triggered test for test_d.ipynb", *cells], AFTER)
    piece = failure_excerpt(job_log(lines), "d/d.ipynb")
    assert piece["error_line"] is None
    assert piece["end"] == line_of(lines, cells[-1]) and piece["end"] - piece["start"] + 1 == EXCERPT_MAX_LINES
    error = ["Kernel died while waiting for execute reply.", *[f"  frame {i}" for i in range(300)]]
    lines = lines_of(STEP, ["Testing notebook: /__w/r/r/notebooks/d/d.ipynb", "Triggered test for test_d.ipynb", *cells, *error], AFTER)
    piece = failure_excerpt(job_log(lines), "d/d.ipynb")
    assert piece["start"] == piece["error_line"] - CONTEXT_BEFORE and piece["end"] - piece["start"] + 1 == EXCERPT_MAX_LINES


def test_section_ends_at_the_next_step():
    lines = lines_of(STEP, ["Testing notebook: /__w/r/r/notebooks/e/e.ipynb", "Start executing cell 0"], AFTER[1:], ["Traceback (most recent call last):"])
    piece = failure_excerpt(job_log(lines), "e/e.ipynb")
    assert piece["error_line"] is None and "Traceback" not in piece["text"]


def test_exact_notebook_path_is_preferred_over_suffix():
    other = ["Testing notebook: /__w/r/r/notebooks/x/b/b.ipynb", "Triggered test for test_b.ipynb", "ERROR in testing test_b.ipynb (other)"]
    lines = lines_of(STEP, other, CELL_ERROR)
    piece = failure_excerpt(job_log(lines), "b/b.ipynb")
    assert "(other)" not in piece["text"] and piece["error_line"] == line_of(lines, CELL_ERROR[8])


def test_not_found_and_without_step_header():
    assert failure_excerpt(job_log(lines_of(STEP, PASSED)), "b/b.ipynb") is None
    assert failure_excerpt(job_log(CELL_ERROR), "b/b.ipynb") is None
    assert failure_excerpt("", "b/b.ipynb") is None
    assert failure_excerpt(None, "b/b.ipynb") is None


def test_excerpt_is_capped():
    lines = lines_of(STEP, ["Testing notebook: /__w/r/r/notebooks/b/b.ipynb", "Traceback (most recent call last):", *["x" * 1000] * 50])
    piece = failure_excerpt(job_log(lines), "b/b.ipynb")
    assert len(piece["text"]) <= EXCERPT_MAX_CHARS + 2 and piece["text"].endswith("…")


def test_file_excerpt_starts_at_the_trigger():
    excerpt = file_excerpt("\n".join(CELL_ERROR[1:])).splitlines()
    assert excerpt[0] == "Triggered test for test_b.ipynb" and "ERROR in testing test_b.ipynb" in excerpt
    assert file_excerpt("only output\nTraceback (most recent call last):").splitlines()[0] == "only output"
    assert file_excerpt("") is None and file_excerpt(None) is None
