import copy
import json

import pytest
from jsonschema import Draft202012Validator

from conftest import FIXTURES_DIR
from nightly_report.metrics import compute_summary

SCHEMA_PATH = FIXTURES_DIR.parent.parent / "schema" / "nightly-report.schema.json"

U10 = "cpu-ubuntu-22.04-py3.10"
U12 = "cpu-ubuntu-22.04-py3.12"
W10 = "cpu-windows-2022-py3.10"
W12 = "cpu-windows-2022-py3.12"

HELLO = "hello-world/hello-world.ipynb"
ASYNC = "async-api/async-api.ipynb"
TFLITE = "tflite-to-openvino/tflite-to-openvino.ipynb"
CHATBOT = "llm-chatbot/llm-chatbot.ipynb"
WHISPER = "whisper-asr/whisper-asr.ipynb"
GPU = "gpu-plugin/gpu-plugin.ipynb"


def counts(planned, executed, passed, failed, timeout, error, skipped, not_run, pass_rate, test_s, compute_s):
    return {
        "planned": planned,
        "executed": executed,
        "passed": passed,
        "failed": failed,
        "timeout": timeout,
        "error": error,
        "skipped": skipped,
        "not_run": not_run,
        "pass_rate": pass_rate,
        "test_s": test_s,
        "compute_s": compute_s,
    }


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def report():
    return json.loads((FIXTURES_DIR / "report_small.json").read_text(encoding="utf-8"))


@pytest.fixture
def summary(report):
    return compute_summary(report)


def empty_report(report):
    empty = copy.deepcopy(report)
    for key in ("configs", "jobs", "notebooks", "results", "failures", "error_groups"):
        empty[key] = []
    empty["run"]["wall_clock_s"] = None
    return empty


def assert_valid(schema, report):
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(report), key=lambda e: list(e.absolute_path))
    assert not errors, "\n".join(f"{list(e.absolute_path)}: {e.message}" for e in errors)


def test_fixture_is_valid_without_summary(schema, report):
    assert "summary" not in report
    schema = copy.deepcopy(schema)
    schema["required"].remove("summary")
    assert_valid(schema, report)


def test_report_with_summary_is_valid(schema, report, summary):
    assert_valid(schema, {**report, "summary": summary})


def test_headline_rates(summary):
    assert summary["pass_rate"] == 0.6
    assert summary["notebook_pass_rate"] == 0.4
    assert summary["coverage"] == 0.75
    assert summary["matrix_coverage"] == 0.625


def test_results_counts(summary):
    assert summary["results"] == counts(24, 15, 9, 3, 2, 1, 4, 5, 0.6, 3912.0, 6620.0)


def test_notebooks(summary):
    assert summary["notebooks"] == {
        "total": 6,
        "tested": 5,
        "untested": 1,
        "untested_list": [GPU],
        "passing_everywhere": 2,
        "failing_somewhere": 3,
        "failing_everywhere": 1,
        "failing_list": [ASYNC, TFLITE, WHISPER],
    }


def test_platform_specific_failures(summary):
    assert summary["platform_specific_failures"] == {"os": {"windows-2022": [ASYNC]}, "python": {"3.12": [TFLITE]}}


def test_by_config(summary):
    assert summary["by_config"] == {
        U10: counts(6, 3, 3, 0, 0, 0, 0, 3, 1.0, 350.0, 1020.0),
        U12: counts(6, 5, 3, 0, 1, 1, 1, 0, 0.6, 2012.0, 2800.0),
        W10: counts(6, 3, 2, 1, 0, 0, 1, 2, 0.6667, 345.0, 800.0),
        W12: counts(6, 4, 1, 2, 1, 0, 2, 0, 0.25, 1205.0, 2000.0),
    }


def test_by_os_and_python(summary):
    assert summary["by_os"] == {
        "ubuntu-22.04": counts(12, 8, 6, 0, 1, 1, 1, 3, 0.75, 2362.0, 3820.0),
        "windows-2022": counts(12, 7, 3, 3, 1, 0, 3, 2, 0.4286, 1550.0, 2800.0),
    }
    assert summary["by_python"] == {
        "3.10": counts(12, 6, 5, 1, 0, 0, 1, 5, 0.8333, 695.0, 1820.0),
        "3.12": counts(12, 9, 4, 2, 2, 1, 3, 0, 0.4444, 3217.0, 4800.0),
    }


def test_infra(summary):
    assert summary["infra"] == {
        "jobs_total": 8,
        "jobs_succeeded": 3,
        "jobs_failed": 4,
        "jobs_cancelled": 1,
        "jobs_failed_before_tests": 1,
        "notebooks_lost": 5,
        "infra_failure_rate": 0.125,
    }


def test_durations(summary):
    durations = summary["durations"]
    assert durations["run_wall_clock_s"] == 17580.0
    assert durations["compute_s"] == 6620.0
    assert durations["test_s"] == 3912.0
    assert durations["setup_overhead_s"] == 2708.0
    # Passed durations: 30, 32, 45, 50, 120, 130, 200, 240, 850.
    assert durations["notebook_median_s"] == 120.0
    assert durations["notebook_p90_s"] == 850.0
    assert durations["notebook_max_s"] == 850.0
    assert durations["near_timeout"] == [{"notebook": CHATBOT, "config_id": U12, "duration_s": 850.0, "timeout_s": 1000.0, "ratio": 0.85}]


def test_slowest(summary):
    slowest = summary["durations"]["slowest"]
    assert len(slowest) == 10
    assert [(e["notebook"], e["config_id"], e["duration_s"]) for e in slowest] == [
        (TFLITE, U12, 1000.0),
        (TFLITE, W12, 1000.0),
        (CHATBOT, U12, 850.0),
        (TFLITE, W10, 240.0),
        (TFLITE, U10, 200.0),
        (ASYNC, U12, 130.0),
        (ASYNC, U10, 120.0),
        (WHISPER, W12, 90.0),
        (ASYNC, W12, 65.0),
        (ASYNC, W10, 60.0),
    ]
    assert slowest[0]["ratio"] == 1.0
    assert slowest[-1] == {"notebook": ASYNC, "config_id": W10, "duration_s": 60.0, "timeout_s": 1000.0, "ratio": 0.06}


def test_failures(summary):
    failures = summary["failures"]
    assert failures["by_category"] == {"cell_error": 3, "timeout": 2, "harness_error": 1}
    assert failures["by_error_class"] == {"dependency": 3, "timeout": 2, "harness": 1}
    assert [(g["signature"], g["occurrences"]) for g in failures["top_error_groups"]] == [("3f2a9c1b7d4e", 3), ("9b8c7d6e5f40", 2), ("0a1b2c3d4e5f", 1)]
    assert failures["top_error_groups"][1] == {
        "signature": "9b8c7d6e5f40",
        "occurrences": 2,
        "exception_type": None,
        "exception_message": "Notebook execution exceeded the timeout of 1000 s",
    }


def test_top_error_groups_limited_to_ten(report):
    group = report["error_groups"][0]
    report["error_groups"] = [{**group, "signature": f"{i:012x}", "occurrences": 20 - i} for i in range(15)]
    top = compute_summary(report)["failures"]["top_error_groups"]
    assert [g["signature"] for g in top] == [f"{i:012x}" for i in range(10)]


def test_openvino_versions(summary):
    assert summary["openvino_versions"] == {"2025.3.0": 13, "2025.4.0.dev20251007": 1, "unknown": 1}
    assert summary["openvino_dev_share"] == round(1 / 14, 4)


def test_summary_key_set(schema, summary):
    assert set(summary) == set(schema["$defs"]["summary"]["properties"])


def test_input_not_mutated(report):
    before = copy.deepcopy(report)
    compute_summary(report)
    assert report == before


def test_empty_report(schema, report):
    empty = empty_report(report)
    summary = compute_summary(empty)
    for rate in ("pass_rate", "notebook_pass_rate", "coverage", "matrix_coverage", "openvino_dev_share"):
        assert summary[rate] is None
    assert summary["results"] == counts(0, 0, 0, 0, 0, 0, 0, 0, None, 0.0, 0.0)
    assert summary["infra"]["infra_failure_rate"] is None
    assert summary["by_config"] == summary["by_os"] == summary["by_python"] == {}
    assert summary["platform_specific_failures"] == {"os": {}, "python": {}}
    assert summary["durations"] == {
        "run_wall_clock_s": None,
        "compute_s": 0.0,
        "test_s": 0.0,
        "setup_overhead_s": 0.0,
        "notebook_median_s": None,
        "notebook_p90_s": None,
        "notebook_max_s": None,
        "slowest": [],
        "near_timeout": [],
    }
    assert summary["openvino_versions"] == {}
    assert_valid(schema, {**empty, "summary": summary})


def test_configs_without_results(schema, report):
    report["results"], report["failures"], report["error_groups"] = [], [], []
    summary = compute_summary(report)
    assert set(summary["by_config"]) == {U10, U12, W10, W12}
    assert summary["by_config"][U10] == counts(0, 0, 0, 0, 0, 0, 0, 0, None, 0.0, 1020.0)
    assert summary["matrix_coverage"] == 0.0
    assert summary["notebooks"]["untested"] == 6
    assert_valid(schema, {**report, "summary": summary})


def test_missing_durations(schema, report):
    for item in report["results"] + report["jobs"]:
        item["duration_s"] = None
    summary = compute_summary(report)
    durations = summary["durations"]
    assert durations["compute_s"] == durations["test_s"] == durations["setup_overhead_s"] == 0.0
    assert durations["notebook_median_s"] is durations["notebook_p90_s"] is durations["notebook_max_s"] is None
    assert durations["slowest"] == durations["near_timeout"] == []
    assert summary["results"]["test_s"] == summary["results"]["compute_s"] == 0.0
    assert_valid(schema, {**report, "summary": summary})


def test_setup_overhead_never_negative(report):
    for job in report["jobs"]:
        job["duration_s"] = 1.0
    assert compute_summary(report)["durations"]["setup_overhead_s"] == 0.0


def test_p90_nearest_rank_and_even_median(report):
    report["results"] = [
        {"notebook": f"nb{i}/nb{i}.ipynb", "config_id": U10, "status": "passed", "duration_s": float(i), "timeout_s": None} for i in range(1, 11)
    ]
    durations = compute_summary(report)["durations"]
    assert durations["notebook_median_s"] == 5.5
    assert durations["notebook_p90_s"] == 9.0
    assert durations["notebook_max_s"] == 10.0
    assert durations["near_timeout"] == []
    assert durations["slowest"][0] == {"notebook": "nb10/nb10.ipynb", "config_id": U10, "duration_s": 10.0, "timeout_s": None, "ratio": None}


def test_near_timeout_threshold_and_order(report):
    def passed(notebook, duration):
        return {"notebook": notebook, "config_id": U10, "status": "passed", "duration_s": duration, "timeout_s": 100}

    report["results"] = [passed("a.ipynb", 79.9), passed("b.ipynb", 80), passed("c.ipynb", 95), {**passed("d.ipynb", 99), "status": "timeout"}]
    near = compute_summary(report)["durations"]["near_timeout"]
    assert [(e["notebook"], e["ratio"]) for e in near] == [("c.ipynb", 0.95), ("b.ipynb", 0.8)]


def test_platform_specific_requires_passing_group(report):
    # whisper-asr fails on both OSes and has no passing results anywhere: never platform-specific.
    summary = compute_summary(report)
    specific = summary["platform_specific_failures"]
    assert WHISPER not in [n for notebooks in specific["os"].values() for n in notebooks]
    assert WHISPER not in [n for notebooks in specific["python"].values() for n in notebooks]


def test_openvino_dev_share_case_insensitive(report):
    for result in report["results"]:
        if result["status"] in ("passed", "failed", "timeout", "error"):
            result["packages"] = {"openvino": "2026.0.0.DEV20261001"}
    summary = compute_summary(report)
    assert summary["openvino_versions"] == {"2026.0.0.DEV20261001": 15}
    assert summary["openvino_dev_share"] == 1.0
