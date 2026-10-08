import copy
import json

import pytest

from conftest import FIXTURES_DIR
from nightly_report.metrics import compute_summary
from nightly_report.summary_md import MAX_FAILURE_ROWS, escape_cell, format_duration, format_rate, render_markdown


@pytest.fixture
def report():
    report = json.loads((FIXTURES_DIR / "report_small.json").read_text(encoding="utf-8"))
    report["summary"] = compute_summary(report)
    return report


@pytest.fixture
def markdown(report):
    return render_markdown(report)


ASYNC_W10 = "cpu-windows-2022-py3.10/async-api/async-api.ipynb"


def failure(report, failure_id):
    return next(f for f in report["failures"] if f["id"] == failure_id)


def section(markdown, title):
    """Text of the '## <title>' section up to the next section."""
    _, _, rest = markdown.partition(f"## {title}")
    assert rest, f"section {title!r} not found"
    return rest.split("\n## ", 1)[0]


def table_row(markdown, first_cell):
    rows = [line for line in markdown.splitlines() if line.startswith(f"| {first_cell} |")]
    assert rows, f"no table row starting with {first_cell!r}"
    return rows[0]


@pytest.mark.parametrize(
    "seconds, expected",
    [(None, "n/a"), (0, "0s"), (45, "45s"), (59.6, "1m 0s"), (725, "12m 5s"), (3600, "1h 0m"), (17580, "4h 53m")],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


@pytest.mark.parametrize("rate, expected", [(None, "n/a"), (0, "0.0%"), (0.6667, "66.7%"), (1.0, "100.0%")])
def test_format_rate(rate, expected):
    assert format_rate(rate) == expected


def test_escape_cell():
    assert escape_cell("a | b\nc\r\n  d") == "a \\| b c d"
    assert escape_cell(None) == ""
    assert escape_cell(3) == "3"


def test_title_and_links(markdown):
    lines = markdown.splitlines()
    assert lines[0] == "# Notebooks nightly — 2026-10-07 (run #412)"
    assert "[Run](https://github.com/openvinotoolkit/openvino_notebooks/actions/runs/18234567890)" in markdown
    assert (
        "commit [`4288ef6`](https://github.com/openvinotoolkit/openvino_notebooks/commit/4288ef68a1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6) "
        "Bump gitpython from 3.1.62 to 3.2.0 in /.docker (#3731)"
    ) in markdown
    assert "event `schedule`" in markdown
    assert "branch `latest`" in markdown


def test_headline_table(markdown):
    assert (
        "| Pass rate | Notebook pass rate | Coverage | Executed | Passed | Failed | Timeout | Error | Not run | Untested notebooks | Wall clock |" in markdown
    )
    assert "| 60.0% | 40.0% | 75.0% | 15 | 9 | 3 | 2 | 1 | 5 | 1 | 4h 53m |" in markdown


def test_by_configuration(markdown):
    text = section(markdown, "By configuration")
    assert "| Config | Executed | Passed | Failed | Timeout | Error | Skipped | Not run | Pass rate | Test time |" in text
    assert table_row(text, "cpu-ubuntu-22.04-py3.10") == "| cpu-ubuntu-22.04-py3.10 | 3 | 3 | 0 | 0 | 0 | 0 | 3 | 100.0% | 5m 50s |"
    assert table_row(text, "cpu-windows-2022-py3.12") == "| cpu-windows-2022-py3.12 | 4 | 1 | 2 | 1 | 0 | 2 | 0 | 25.0% | 20m 5s |"


def test_failures_table(markdown):
    text = section(markdown, "Failures (6)")
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert len(rows) == 6
    # Sorted by notebook, then config.
    assert rows[0].startswith("| `async-api/async-api.ipynb` | cpu-windows-2022-py3.10 | failed | cell_error / dependency |")
    assert rows[-1].startswith("| `whisper-asr/whisper-asr.ipynb` | cpu-windows-2022-py3.12 |")
    assert "ImportError: DLL load failed while importing _speedups: The specified module could not be found." in rows[0]
    assert "[job](https://github.com/openvinotoolkit/openvino_notebooks/actions/runs/18234567890/job/31000000005)" in rows[0]
    assert "[logs](https://github.com/openvinotoolkit/openvino_notebooks/actions/runs/18234567890/artifacts/31000001005)" in rows[0]
    assert "[source](" in rows[0]


def test_failure_message_pipes_escaped_and_single_line(markdown):
    text = section(markdown, "Failures (6)")
    harness = table_row(text, "`whisper-asr/whisper-asr.ipynb` | cpu-ubuntu-22.04-py3.12")
    assert "| error | harness_error / harness | HarnessError: Failed to create notebook venv: pip install exited with code 1 \\| retries disabled |" in harness
    # The harness failure has no logs artifact: only job and source links.
    assert "[logs]" not in harness
    assert harness.count(" · ") == 1
    timeout = table_row(text, "`tflite-to-openvino/tflite-to-openvino.ipynb` | cpu-ubuntu-22.04-py3.12")
    assert "| Notebook execution exceeded the timeout of 1000 s |" in timeout


def test_long_error_truncated(report):
    failure(report, ASYNC_W10)["exception_message"] = "x" * 500
    row = table_row(render_markdown(report), "`async-api/async-api.ipynb` | cpu-windows-2022-py3.10")
    error = row.split(" | ")[4]
    assert error.startswith("ImportError: xxx")
    assert error.endswith("…")
    assert len(error) == 120


def test_failures_limited(report):
    first = report["failures"][0]
    report["failures"] = [{**first, "notebook": f"nb{i:03d}/nb{i:03d}.ipynb"} for i in range(MAX_FAILURE_ROWS + 7)]
    text = section(render_markdown(report), f"Failures ({MAX_FAILURE_ROWS + 7})")
    assert len([line for line in text.splitlines() if line.startswith("| `nb")]) == MAX_FAILURE_ROWS
    assert "…and 7 more" in text


def test_error_groups(markdown, report):
    text = section(markdown, "Error groups")
    assert "| Signature | Occurrences | Error | Notebooks |" in text
    assert (
        "| `3f2a9c1b7d4e` | 3 | ImportError: DLL load failed while importing _speedups: The specified module could not be found. | "
        "`async-api/async-api.ipynb`, `whisper-asr/whisper-asr.ipynb` |"
    ) in text
    assert "retries disabled" in text and "\\| retries disabled" in text


def test_error_group_notebooks_limited(report):
    report["error_groups"][0]["notebooks"] = [f"nb{i}/nb{i}.ipynb" for i in range(8)]
    text = section(render_markdown(report), "Error groups")
    assert "`nb4/nb4.ipynb` +3 |" in text
    assert "nb5/nb5.ipynb" not in text


def test_infrastructure(markdown):
    text = section(markdown, "Infrastructure")
    assert (
        "- [ubuntu-22.04 / python 3.10 / cpu (batch_1)](https://github.com/openvinotoolkit/openvino_notebooks/actions/runs/18234567890/job/31000000002)"
        " — failed step: Install python dependencies"
    ) in text
    assert "(not run): 5" in text


def test_near_timeout_and_platform_specific(markdown):
    near = section(markdown, "Near timeout")
    assert "| `llm-chatbot/llm-chatbot.ipynb` | cpu-ubuntu-22.04-py3.12 | 14m 10s | 16m 40s | 85.0% |" in near
    specific = section(markdown, "Platform-specific failures")
    assert "- OS `windows-2022`: `async-api/async-api.ipynb`" in specific
    assert "- Python `3.12`: `tflite-to-openvino/tflite-to-openvino.ipynb`" in specific


def test_optional_sections_omitted(report):
    for job in report["jobs"]:
        job["tests_started"] = True
    for result in report["results"]:
        if result["status"] == "not_run":
            result["status"] = "skipped"
    report["summary"] = compute_summary(report)
    report["summary"]["durations"]["near_timeout"] = []
    report["summary"]["platform_specific_failures"] = {"os": {}, "python": {}}
    markdown = render_markdown(report)
    for title in ("Infrastructure", "Near timeout", "Platform-specific failures"):
        assert f"## {title}" not in markdown
    assert "## Failures (6)" in markdown


def test_empty_report_renders(report):
    empty = copy.deepcopy(report)
    for key in ("configs", "jobs", "notebooks", "results", "failures", "error_groups"):
        empty[key] = []
    empty["run"]["wall_clock_s"] = None
    empty["summary"] = compute_summary(empty)
    markdown = render_markdown(empty)
    assert markdown.startswith("# Notebooks nightly — 2026-10-07 (run #412)\n")
    assert "| n/a | n/a | n/a | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |" in markdown
    assert "## Failures (0)" in markdown
    for title in ("Error groups", "Infrastructure", "Near timeout", "Platform-specific failures"):
        assert f"## {title}" not in markdown


def test_bare_report_renders():
    markdown = render_markdown({})
    assert markdown.startswith("# Notebooks nightly — unknown date\n")
    assert "n/a" in markdown


def test_missing_optional_fields(report):
    run = report["run"]
    run["number"] = None
    run["commit"] = {"sha": "0123456789abcdef", "short_sha": None, "message": None, "url": None}
    for failure in report["failures"]:
        failure["exception_type"] = None
        failure["exception_message"] = None
        failure["links"] = {}
    for group in report["error_groups"]:
        group.pop("exception_type", None)
        group["exception_message"] = None
    for job in report["jobs"]:
        job["failed_step"] = None
    report["summary"]["durations"]["run_wall_clock_s"] = None
    report["summary"]["durations"]["near_timeout"][0]["timeout_s"] = None
    report["summary"]["durations"]["near_timeout"][0]["ratio"] = None

    markdown = render_markdown(report)
    assert markdown.splitlines()[0] == "# Notebooks nightly — 2026-10-07"
    assert "commit `0123456` · event" in markdown
    assert "failed step: unknown" in markdown
    assert table_row(markdown, "`async-api/async-api.ipynb` | cpu-windows-2022-py3.10").endswith("| failed | cell_error / dependency |  |  |")


def test_summary_computed_when_missing(report):
    expected = render_markdown(report)
    del report["summary"]
    assert render_markdown(report) == expected


def test_html_like_text_is_escaped(report):
    failure(report, ASYNC_W10)["exception_message"] = "bad value <class 'int'>"
    row = table_row(render_markdown(report), "`async-api/async-api.ipynb` | cpu-windows-2022-py3.10")
    assert "bad value &lt;class 'int'&gt;" in row
