"""Unit tests of the nightly failure collector (pre-agent step of the nightly issue remediator); no network, no real `gh`."""

import argparse
import subprocess
import types
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import collector_common as common
import collector_github as github
import collector_analysis as analysis
import collector_evidence as evidence
import collector_render as render
import collect_nightly_failures as cnf
from nightly_builders import (
    CPU_CONFIG,
    CPU_CONFIG_312,
    GPU_CONFIG,
    INFRA_REASON,
    INFRA_STEP,
    REPO,
    FakeGh,
    make_failure,
    make_job,
    make_report,
    make_result,
    make_run,
    make_zip,
    nightly_report,
)

NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
SIG = "aaaaaaaaaaaa"


def make_group(signature, kind="error", error_class="other", notebooks=1, configs=1):
    group = common.Group(signature, kind, "cell_error", error_class, "Error", "boom")
    group.members = [common.Member(f"{signature}-{n}/nb.ipynb", f"cfg{c}", "failed") for n in range(notebooks) for c in range(configs)]
    return group


def make_tracked(signature, state="open", kind="pr", days_ago=None, number=1):
    closed_at = (NOW - timedelta(days=days_ago)).isoformat() if days_ago is not None else None
    return common.TrackedItem(kind, number, f"https://example.test/{number}", f"title (sig {signature})", state, signature, "2026-01-01T00:00:00Z", closed_at)


def make_night(number, signatures=None, statuses=None):
    info = {
        "number": number,
        "url": f"https://example.test/runs/{number}",
        "nightly_date": f"2026-01-{number:02d}",
        "conclusion": "failure",
        "commit": {"short_sha": f"c{number}"},
        "inputs": {"docker_tag": f"tag{number}"},
    }
    return common.Night(info=info, signatures=signatures or {}, statuses=statuses or {})


def make_args(**overrides):
    values = dict(repo=REPO, source_repo=REPO, run_id="", max_groups=5, max_prs=3, history_nights=3, logs_per_group=3, out="/tmp/nightly/digest")
    values.update(overrides)
    return argparse.Namespace(**values)


# ----------------------------------------------------------------------------------------------------------------------
# gh wrapper
# ----------------------------------------------------------------------------------------------------------------------


class FakeRun:
    """Stands in for `subprocess.run`: plays back (returncode, stdout, stderr) tuples or raises exceptions."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        returncode, stdout, stderr = response
        handle = kwargs.get("stdout")
        if handle is not None and handle is not subprocess.PIPE:
            handle.write(stdout)
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def fake_run(monkeypatch):
    def install(*responses):
        fake = FakeRun(*responses)
        monkeypatch.setattr(github.subprocess, "run", fake)
        monkeypatch.setattr(github.time, "sleep", lambda seconds: None)
        return fake

    return install


def test_gh_returns_stdout(fake_run):
    fake = fake_run((0, b"out", b""))
    assert github.gh(["api", "x"]) == b"out"
    assert fake.calls == [["gh", "api", "x"]]


def test_gh_streams_stdout_to_a_file(fake_run, tmp_path):
    fake_run((0, b"zip data", b""))
    target = tmp_path / "artifact.zip"
    assert github.gh(["api", "x"], stdout_path=target) == b""
    assert target.read_bytes() == b"zip data"


def test_gh_retries_transient_errors(fake_run):
    fake = fake_run((1, b"", b"HTTP 502: Bad Gateway"), subprocess.TimeoutExpired(["gh"], 1), (0, b"ok", b""))
    assert github.gh(["api", "x"]) == b"ok"
    assert len(fake.calls) == 3


def test_gh_gives_up_after_the_retries(fake_run):
    fake = fake_run(*[(1, b"", b"HTTP 503: unavailable")] * 3)
    with pytest.raises(common.GhError, match="HTTP 503"):
        github.gh(["api", "x"])
    assert len(fake.calls) == 3


def test_gh_does_not_retry_permanent_errors(fake_run):
    fake = fake_run((1, b"", b"HTTP 404: Not Found"))
    with pytest.raises(common.GhError) as error:
        github.gh(["api", "x"])
    assert error.value.returncode == 1 and "gh api x failed (1)" in str(error.value)
    assert len(fake.calls) == 1


def test_gh_allows_escape_sequences_for_job_logs(fake_run):
    refusal = b"the response contains terminal escape sequences; pass --allow-escape-sequences to print it"
    fake = fake_run((1, b"", refusal), (0, b"log", b""))
    assert github.gh(["api", "repos/a/b/actions/jobs/1/logs"]) == b"log"
    assert fake.calls == [["gh", "api", "repos/a/b/actions/jobs/1/logs"], ["gh", "api", "--allow-escape-sequences", "repos/a/b/actions/jobs/1/logs"]]


def test_gh_asks_for_escape_sequences_only_once_and_only_for_api(fake_run):
    refusal = b"pass --allow-escape-sequences"
    fake = fake_run((1, b"", refusal), (1, b"", refusal), (1, b"", refusal))
    with pytest.raises(common.GhError):
        github.gh(["api", "x"])
    assert len(fake.calls) == 2
    fake = fake_run((1, b"", refusal))
    with pytest.raises(common.GhError):
        github.gh(["pr", "list"])
    assert len(fake.calls) == 1


# ----------------------------------------------------------------------------------------------------------------------
# Zip and artifact helpers
# ----------------------------------------------------------------------------------------------------------------------


def test_read_zip_member(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(make_zip({"logs/a.log": "hello", "big.bin": "x" * 100}))
    assert github.read_zip_member(archive, "logs/a.log", 1000) == b"hello"
    assert github.read_zip_member(archive, "missing", 1000) is None
    with pytest.raises(common.CollectorError, match="larger than 10"):
        github.read_zip_member(archive, "big.bin", 10)


def test_download_artifact_rejects_non_zip_data(monkeypatch, tmp_path):
    monkeypatch.setattr(github, "gh", lambda args, stdout_path=None, **kwargs: stdout_path.write_bytes(b"<html>not a zip</html>"))
    with pytest.raises(common.CollectorError, match="not a zip"):
        github.download_artifact(REPO, 5, tmp_path / "x.zip")


def test_find_artifact_takes_the_newest_unexpired_one_by_name(monkeypatch):
    artifacts = [
        {"id": 1, "name": common.REPORT_ARTIFACT, "expired": False},
        {"id": 5, "name": common.REPORT_ARTIFACT, "expired": True},
        {"id": 3, "name": common.REPORT_ARTIFACT, "expired": False},
        {"id": 9, "name": "something-else", "expired": False},
    ]
    monkeypatch.setattr(github, "api_json", lambda path: {"artifacts": artifacts})
    assert github.find_artifact(REPO, 1, common.REPORT_ARTIFACT)["id"] == 3
    monkeypatch.setattr(github, "api_json", lambda path: {"artifacts": artifacts[1:2]})
    assert github.find_artifact(REPO, 1, common.REPORT_ARTIFACT) is None


def test_parse_artifact_id_and_safe_name():
    assert evidence.parse_artifact_id("https://github.com/a/b/actions/runs/1/artifacts/13646794134") == 13646794134
    assert evidence.parse_artifact_id("https://github.com/a/b/actions/runs/1") is None
    assert evidence.parse_artifact_id(None) is None
    assert evidence.safe_name("a/b c.ipynb") == "a-b-c.ipynb"
    assert evidence.safe_name("///") == "unnamed"
    assert len(evidence.safe_name("x" * 500)) == 120


def test_parse_ts():
    assert analysis.parse_ts("2026-01-10T00:00:00Z") == datetime(2026, 1, 10, tzinfo=timezone.utc)
    assert analysis.parse_ts("2026-01-10T00:00:00") == datetime(2026, 1, 10, tzinfo=timezone.utc)
    assert analysis.parse_ts("2026-01-10T01:00:00+01:00") == datetime(2026, 1, 10, tzinfo=timezone.utc)
    assert analysis.parse_ts("") is None and analysis.parse_ts(None) is None and analysis.parse_ts("yesterday") is None


def test_normalize_log_strips_ansi_and_carriage_returns():
    assert evidence.normalize_log(b"\x1b[31mred\x1b[0m\r\nnext\rlast") == "red\nnext\nlast"


# ----------------------------------------------------------------------------------------------------------------------
# Run resolution and report schema
# ----------------------------------------------------------------------------------------------------------------------


def test_resolve_run_picks_the_latest_finished_nightly(monkeypatch):
    runs = [{"id": 3, "conclusion": "cancelled"}, {"id": 2, "conclusion": "failure"}, {"id": 1, "conclusion": "success"}]
    monkeypatch.setattr(github, "list_nightly_runs", lambda repo, per_page=30: runs)
    assert github.resolve_run(REPO, "")["id"] == 2
    monkeypatch.setattr(github, "list_nightly_runs", lambda repo, per_page=30: runs[:1])
    with pytest.raises(common.NoWork):
        github.resolve_run(REPO, "")


def test_resolve_run_checks_an_explicit_run(monkeypatch):
    run = {"id": 7, "path": ".github/workflows/treon_nightly.yml", "status": "completed"}
    monkeypatch.setattr(github, "api_json", lambda path: run)
    assert github.resolve_run(REPO, "7") is run
    monkeypatch.setattr(github, "api_json", lambda path: {**run, "path": ".github/workflows/codecheck.yml"})
    with pytest.raises(common.CollectorError, match="not a treon_nightly.yml run"):
        github.resolve_run(REPO, "7")
    monkeypatch.setattr(github, "api_json", lambda path: {**run, "status": "in_progress"})
    with pytest.raises(common.NoWork, match="not completed"):
        github.resolve_run(REPO, "7")


def test_check_schema():
    github.check_schema(make_report())
    github.check_schema(make_report(schema_version="1.9.3"))
    for version in ("2.0.0", "", "x.y"):
        with pytest.raises(common.NoWork, match="unsupported"):
            github.check_schema(make_report(schema_version=version))
    report = make_report()
    del report["jobs"]
    with pytest.raises(common.NoWork, match="`jobs`"):
        github.check_schema(report)


# ----------------------------------------------------------------------------------------------------------------------
# Error groups and ranking
# ----------------------------------------------------------------------------------------------------------------------


def test_build_groups_makes_error_and_infra_groups():
    report = make_report(
        failures=[
            make_failure("a/a.ipynb", "cfg1", SIG, "dependency"),
            make_failure("a/a.ipynb", "cfg2", SIG, "dependency"),
            make_failure("b/b.ipynb", "cfg1", "bbbbbbbbbbbb"),
            {**make_failure("c/c.ipynb", "cfg1", "cccccccccccc"), "signature": None},
        ],
        results=[
            make_result("d/d.ipynb", "cfg3", "not_run", job_id=13, not_run_reason=INFRA_REASON),
            make_result("e/e.ipynb", "cfg3", "not_run", job_id=13, not_run_reason=INFRA_REASON),
            make_result("e/e.ipynb", "cfg4", "not_run", not_run_reason="no_fragment"),
            make_result("f/f.ipynb", "cfg3", "not_run", not_run_reason="job_interrupted"),
            make_result("g/g.ipynb", "cfg3", "not_run", not_run_reason="job_cancelled"),
            make_result("h/h.ipynb", "cfg3", "not_run", not_run_reason="job_skipped"),
            make_result("i/i.ipynb", "cfg3", "failed", not_run_reason=INFRA_REASON),
        ],
    )
    groups = {g.signature: g for g in analysis.build_groups(report)}
    infra = analysis.infra_signature(INFRA_REASON)
    assert set(groups) == {SIG, "bbbbbbbbbbbb", infra, analysis.infra_signature("no_fragment"), analysis.infra_signature("job_interrupted")}
    assert groups[SIG].kind == "error" and groups[SIG].notebooks == ["a/a.ipynb"] and groups[SIG].configs == ["cfg1", "cfg2"]
    assert groups[infra].kind == "infra" and groups[infra].notebooks == ["d/d.ipynb", "e/e.ipynb"]
    assert groups[infra].headline == f"job_failed_before_tests: {INFRA_STEP}"
    assert groups[analysis.infra_signature("no_fragment")].headline == "no_fragment"
    assert all(m.status == "not_run" and m.not_run_reason for m in groups[infra].members)


def test_build_groups_prefers_the_group_header_of_the_report():
    report = make_report(
        failures=[make_failure("a/a.ipynb", "cfg1", SIG, "other", message="member message", exception_type="ValueError")],
        error_groups=[{"signature": SIG, "category": "cell_error", "error_class": "dependency", "exception_type": None, "exception_message": "group message"}],
    )
    (group,) = analysis.build_groups(report)
    assert (group.error_class, group.exception_message, group.exception_type) == ("dependency", "group message", "ValueError")
    assert group.headline == "ValueError: group message"


def test_rank_key_prefers_more_notebooks_then_more_pairs():
    wide, deep, small = make_group("w", notebooks=3), make_group("d", notebooks=2, configs=3), make_group("s", notebooks=2)
    assert [g.signature for g in sorted([small, deep, wide], key=analysis.rank_key)] == ["w", "d", "s"]


def test_pr_candidates_are_error_groups_that_a_change_can_fix():
    assert all(analysis.is_pr_candidate(make_group("x", error_class=name)) for name in ("dependency", "assertion", "openvino", "other"))
    assert not any(analysis.is_pr_candidate(make_group("x", error_class=name)) for name in ("timeout", "memory", "disk", "crash", "network", "harness"))
    assert not analysis.is_pr_candidate(make_group("x", kind="infra", error_class="other"))


# ----------------------------------------------------------------------------------------------------------------------
# Tracked PRs and issues
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Error signature: `0123456789ab`", "0123456789ab"),
        ("**Error signature:** 0123456789AB", "0123456789ab"),
        ("[nightly-fix] Something broke (sig 0123456789ab)", "0123456789ab"),
        ("Error signature:\n\n`0123456789ab`", "0123456789ab"),
        ("signature 0123456789ab", None),
        ("Error signature: 0123456789abc", None),
        ("Error signature: not-a-signature", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_signature(text, expected):
    assert analysis.extract_signature(text) == expected


def test_extract_signature_prefers_the_title():
    assert analysis.extract_signature("x (sig aaaaaaaaaaaa)", "Error signature: bbbbbbbbbbbb") == "aaaaaaaaaaaa"
    assert analysis.extract_signature("no signature here", "Error signature: bbbbbbbbbbbb") == "bbbbbbbbbbbb"


@pytest.mark.parametrize(
    "kind, raw, expected",
    [
        ("pr", {"state": "OPEN"}, "open"),
        ("pr", {"state": "MERGED"}, "merged"),
        ("pr", {"state": "CLOSED", "mergedAt": "2026-01-01T00:00:00Z"}, "merged"),
        ("pr", {"state": "CLOSED"}, "closed"),
        ("issue", {"state": "OPEN"}, "open"),
        ("issue", {"state": "CLOSED", "stateReason": "NOT_PLANNED"}, "not_planned"),
        ("issue", {"state": "CLOSED", "stateReason": "DUPLICATE"}, "duplicate"),
        ("issue", {"state": "CLOSED", "stateReason": "COMPLETED"}, "completed"),
        ("issue", {"state": "CLOSED"}, "completed"),
    ],
)
def test_item_state(kind, raw, expected):
    assert analysis.item_state(kind, raw) == expected


def test_fetch_tracked_reads_signatures_from_titles_and_bodies(monkeypatch):
    fake = FakeGh()
    fake.prs = [
        {"number": 1, "title": "[nightly-fix] a (sig aaaaaaaaaaaa)", "body": "", "state": "OPEN", "url": "u1", "createdAt": "2026-01-02T00:00:00Z"},
        {
            "number": 2,
            "title": "fix",
            "body": "Error signature: `aaaaaaaaaaaa`",
            "state": "MERGED",
            "url": "u2",
            "createdAt": "2026-01-03T00:00:00Z",
            "updatedAt": "2026-01-04T00:00:00Z",
            "mergedAt": "2026-01-04T00:00:00Z",
        },
        {"number": 3, "title": "unrelated", "body": "", "state": "OPEN", "url": "u3", "createdAt": "2026-01-03T00:00:00Z"},
    ]
    fake.issues = [
        {
            "number": 4,
            "title": "[nightly] b",
            "body": "Error signature: bbbbbbbbbbbb",
            "state": "CLOSED",
            "stateReason": "NOT_PLANNED",
            "url": "u4",
            "createdAt": "2026-01-05T00:00:00Z",
            "updatedAt": "2026-01-06T00:00:00Z",
            "closedAt": "2026-01-06T00:00:00Z",
        }
    ]
    monkeypatch.setattr(github, "gh", fake)
    warnings = []
    tracked = analysis.fetch_tracked(REPO, warnings.append, NOW)
    assert set(tracked) == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert [(i.label, i.state) for i in tracked["aaaaaaaaaaaa"]] == [("PR #2", "merged"), ("PR #1", "open")]
    assert [(i.label, i.state) for i in tracked["bbbbbbbbbbbb"]] == [("issue #4", "not_planned")]
    assert warnings == []
    assert fake.count("--raw-field label=nightly-remediator") == 4
    assert fake.count("states:[OPEN]") == 2
    assert fake.count("states:[CLOSED") == 2


def test_fetch_tracked_tolerates_disabled_issues_only(monkeypatch):
    fake = FakeGh()
    fake.prs = [{"number": 1, "title": "x (sig aaaaaaaaaaaa)", "body": "", "state": "OPEN", "url": "u", "createdAt": "2026-01-02T00:00:00Z"}]
    fake.issues_disabled = True
    monkeypatch.setattr(github, "gh", fake)
    warnings = []
    assert list(analysis.fetch_tracked(REPO, warnings.append)) == ["aaaaaaaaaaaa"]
    assert len(warnings) == 1 and "disabled" in warnings[0]

    fake.issues_disabled = False
    fake.fail_on = "issues(first:100"
    with pytest.raises(common.GhError):
        analysis.fetch_tracked(REPO, warnings.append)


@pytest.mark.parametrize("cursors", [("same", "same"), ("first", "second", "first")])
def test_tracking_rejects_repeated_pagination_cursors(monkeypatch, cursors):
    item = {"number": 1, "title": "x", "body": "", "state": "OPEN", "createdAt": "2026-01-02T00:00:00Z"}
    cursor_iter = iter(cursors)

    def repeated_page(args):
        return {
            "data": {
                "repository": {
                    "hasIssuesEnabled": True,
                    "pullRequests": {"nodes": [item], "pageInfo": {"hasNextPage": True, "endCursor": next(cursor_iter)}},
                }
            }
        }

    monkeypatch.setattr(analysis, "list_json", repeated_page)
    with pytest.raises(common.CollectorError, match="invalid pr tracking cursor"):
        analysis.fetch_tracked(REPO, lambda message: None, NOW)


def test_tracking_rejects_an_invalid_item_state(monkeypatch):
    def invalid_page(args):
        return {
            "data": {
                "repository": {
                    "hasIssuesEnabled": True,
                    "pullRequests": {"nodes": [{"number": 1, "state": "UNKNOWN"}], "pageInfo": {"hasNextPage": False, "endCursor": None}},
                }
            }
        }

    monkeypatch.setattr(analysis, "list_json", invalid_page)
    with pytest.raises(common.CollectorError, match="invalid pr tracking item"):
        analysis.fetch_tracked(REPO, lambda message: None, NOW)


@pytest.mark.parametrize("field", ["updatedAt", "closedAt"])
def test_tracking_rejects_invalid_timestamps_on_rejected_items(monkeypatch, field):
    fake = FakeGh()
    item = {
        "number": 1,
        "title": "x (sig aaaaaaaaaaaa)",
        "body": "",
        "state": "CLOSED",
        "createdAt": "2026-01-02T00:00:00Z",
        "updatedAt": "2026-01-04T00:00:00Z",
        "closedAt": "2026-01-04T00:00:00Z",
    }
    item[field] = "invalid"
    fake.prs = [item]
    monkeypatch.setattr(github, "gh", fake)

    with pytest.raises(common.CollectorError, match=field):
        analysis.fetch_tracked(REPO, lambda message: None, NOW)


def test_open_items_block_a_group():
    decision = analysis.decide_tracking([make_tracked(SIG, "merged", days_ago=2, number=1), make_tracked(SIG, "open", number=2)], NOW)
    assert decision.skip and decision.reason == "open" and decision.blocking.number == 2
    assert [i.number for i in decision.previous] == [1]


@pytest.mark.parametrize("state", ["closed", "not_planned", "duplicate"])
def test_recently_rejected_items_block_a_group(state):
    decision = analysis.decide_tracking([make_tracked(SIG, state, days_ago=10)], NOW)
    assert decision.skip and decision.reason == "recently rejected"


@pytest.mark.parametrize(
    "state, days_ago", [("closed", 30), ("closed", 31), ("not_planned", 30), ("not_planned", 90), ("duplicate", 30), ("merged", 1), ("completed", 1)]
)
def test_old_rejections_and_finished_work_do_not_block_but_are_passed_on(state, days_ago):
    decision = analysis.decide_tracking([make_tracked(SIG, state, days_ago=days_ago)], NOW)
    assert not decision.skip and decision.blocking is None
    assert [i.state for i in decision.previous] == [state]


def test_untracked_groups_are_not_blocked():
    decision = analysis.decide_tracking([], NOW)
    assert not decision.skip and decision.previous == []


def test_previous_attempts_are_capped():
    items = [make_tracked(SIG, "merged", days_ago=40 + n, number=n) for n in range(8)]
    assert len(analysis.decide_tracking(items, NOW).previous) == 5


# ----------------------------------------------------------------------------------------------------------------------
# Selection of groups for the agent
# ----------------------------------------------------------------------------------------------------------------------


def signatures(items):
    return [(item[0] if isinstance(item, tuple) else item).signature for item in items]


def test_selection_gives_pr_candidates_the_first_slots():
    groups = [
        make_group("f1", error_class="dependency", notebooks=3),
        make_group("f2", error_class="assertion", notebooks=2),
        make_group("f3", error_class="other"),
        make_group("f4", error_class="openvino"),
        make_group("i1", kind="infra", error_class="no_fragment", notebooks=10),
        make_group("t1", error_class="timeout", notebooks=6),
        make_group("c1", error_class="crash", notebooks=4),
    ]
    selected, skipped, deferred = analysis.select_groups(groups, {}, max_groups=5, max_prs=3, now=NOW)
    assert signatures(selected) == ["f1", "f2", "f3", "i1", "t1"]
    assert signatures(deferred) == ["f4", "c1"]
    assert skipped == []


def test_selection_fills_free_slots_with_the_other_kind():
    few_fixable = [make_group("f1", error_class="dependency")] + [make_group(f"t{n}", error_class="timeout", notebooks=10 - n) for n in range(6)]
    selected, _, deferred = analysis.select_groups(few_fixable, {}, max_groups=5, max_prs=3, now=NOW)
    assert signatures(selected) == ["f1", "t0", "t1", "t2", "t3"] and signatures(deferred) == ["t4", "t5"]

    only_fixable = [make_group(f"f{n}", error_class="other", notebooks=10 - n) for n in range(7)]
    selected, _, deferred = analysis.select_groups(only_fixable, {}, max_groups=5, max_prs=3, now=NOW)
    assert signatures(selected) == ["f0", "f1", "f2", "f3", "f4"] and signatures(deferred) == ["f5", "f6"]

    mixed = [make_group(f"f{n}", error_class="other", notebooks=10 - n) for n in range(3)] + [make_group("t0", error_class="timeout")]
    selected, _, _ = analysis.select_groups(mixed, {}, max_groups=5, max_prs=1, now=NOW)
    assert signatures(selected) == ["f0", "f1", "f2", "t0"]


def test_selection_never_exceeds_the_group_cap():
    groups = [make_group(f"f{n}", error_class="other", notebooks=9 - n) for n in range(4)] + [make_group(f"t{n}", error_class="timeout") for n in range(3)]
    selected, _, deferred = analysis.select_groups(groups, {}, max_groups=2, max_prs=10, now=NOW)
    assert signatures(selected) == ["f0", "f1"] and len(deferred) == 5
    selected, _, _ = analysis.select_groups(groups, {}, max_groups=1, max_prs=1, now=NOW)
    assert signatures(selected) == ["f0"]


def test_tracked_groups_do_not_use_a_slot():
    groups = [
        make_group("f1", error_class="dependency", notebooks=5),
        make_group("f2", error_class="other", notebooks=4),
        make_group("t1", error_class="timeout"),
    ]
    tracked = {"f1": [make_tracked("f1", "open")], "f2": [make_tracked("f2", "closed", days_ago=3, number=2)]}
    selected, skipped, deferred = analysis.select_groups(groups, tracked, max_groups=1, max_prs=3, now=NOW)
    assert signatures(selected) == ["t1"] and signatures(skipped) == ["f1", "f2"] and deferred == []
    assert [d.reason for _, d in skipped] == ["open", "recently rejected"]


def test_selected_groups_carry_the_earlier_attempts():
    selected, _, _ = analysis.select_groups([make_group("f1")], {"f1": [make_tracked("f1", "merged", days_ago=5)]}, max_groups=5, max_prs=3, now=NOW)
    assert [i.state for i in selected[0][1].previous] == ["merged"]


# ----------------------------------------------------------------------------------------------------------------------
# History of previous nightlies
# ----------------------------------------------------------------------------------------------------------------------


def test_trend_label():
    has, lacks = make_night(9, {SIG: {("a", "c")}}), make_night(8)
    assert analysis.trend_label(SIG, []) == "unknown (no history)"
    assert analysis.trend_label(SIG, [lacks, lacks]) == "new tonight"
    assert analysis.trend_label(SIG, [lacks, has]) == "intermittent (2/3 nights)"
    assert analysis.trend_label(SIG, [has, has]) == "recurring (all 3 nights)"
    assert analysis.trend_label(SIG, [has, lacks]) == "recurring (2/3 nights)"


def test_history_rows_classify_every_pair_of_the_group():
    group = make_group(SIG)
    group.members = [common.Member(name, "c1", "failed") for name in ("a", "b", "c", "d")]
    keys = {m.key for m in group.members}
    current = make_night(10, {SIG: keys})
    previous = make_night(9, {SIG: {("a", "c1")}}, {("a", "c1"): "failed", ("b", "c1"): "passed", ("c", "c1"): "timeout"})
    rows = analysis.history_rows(group, current, [previous])
    assert [r["current"] for r in rows] == [True, False]
    assert (rows[0]["same_error"], rows[0]["passed"]) == (4, 0)
    assert (rows[1]["same_error"], rows[1]["passed"], rows[1]["other_failure"], rows[1]["not_run_or_absent"]) == (1, 1, 1, 1)
    assert (rows[1]["date"], rows[1]["commit"], rows[1]["docker_tag"], rows[1]["run_number"]) == ("2026-01-09", "c9", "tag9", 9)


def test_night_from_report_keeps_signatures_and_statuses():
    night = analysis.night_from_report(nightly_report())
    assert ("a/a.ipynb", CPU_CONFIG) in night.signatures["aaaaaaaaaaaa"]
    assert night.statuses[("g/g.ipynb", CPU_CONFIG)] == "passed"
    assert night.info["id"] == 1001


def test_fetch_history_uses_older_finished_nights_newest_first(monkeypatch, tmp_path):
    runs = [
        {"id": 5, "created_at": "2026-01-12T00:00:00Z", "conclusion": "failure"},
        {"id": 4, "created_at": "2026-01-10T00:00:00Z", "conclusion": "failure"},
        {"id": 3, "created_at": "2026-01-09T00:00:00Z", "conclusion": "cancelled"},
        {"id": 2, "created_at": "2026-01-08T00:00:00Z", "conclusion": "success"},
        {"id": 1, "created_at": "2026-01-07T00:00:00Z", "conclusion": "failure"},
        {"id": 0, "created_at": "2026-01-06T00:00:00Z", "conclusion": "failure"},
    ]
    downloads = []

    def fake_download(repo, run_id, workdir, with_environments):
        downloads.append(run_id)
        if run_id == 2:
            raise common.NoWork("artifact expired")
        return make_report(run=make_run(run_id, run_id)), {}

    monkeypatch.setattr(analysis, "list_nightly_runs", lambda repo, per_page=30: runs)
    monkeypatch.setattr(analysis, "download_report", fake_download)
    warnings = []
    nights = analysis.fetch_history(REPO, runs[1], 2, tmp_path, warnings.append)
    assert [n.info["id"] for n in nights] == [1, 0]
    assert downloads == [2, 1, 0]
    assert len(warnings) == 1 and "run 2" in warnings[0]


def test_fetch_history_is_best_effort(monkeypatch, tmp_path):
    def unavailable(repo, per_page=30):
        raise common.GhError(["api"], 1, "HTTP 500: server error")

    warnings = []
    monkeypatch.setattr(analysis, "list_nightly_runs", unavailable)
    assert analysis.fetch_history(REPO, {"id": 4, "created_at": "2026-01-10T00:00:00Z"}, 3, tmp_path, warnings.append) == []
    assert len(warnings) == 1 and "history unavailable" in warnings[0]

    monkeypatch.setattr(analysis, "list_nightly_runs", lambda *args, **kwargs: pytest.fail("history was not requested"))
    assert analysis.fetch_history(REPO, {"id": 4, "created_at": "2026-01-10T00:00:00Z"}, 0, tmp_path, warnings.append) == []


# ----------------------------------------------------------------------------------------------------------------------
# Evidence
# ----------------------------------------------------------------------------------------------------------------------


def test_pick_representatives_prefers_distinct_notebooks_then_distinct_configs():
    group = make_group("x")
    group.members = [
        common.Member("a.ipynb", "dgpu-ubuntu-22.04-py3.11", "failed"),
        common.Member("a.ipynb", "cpu-ubuntu-22.04-py3.12", "failed"),
        common.Member("a.ipynb", "cpu-ubuntu-22.04-py3.11", "failed"),
        common.Member("b.ipynb", "dgpu-ubuntu-22.04-py3.11", "failed"),
        common.Member("b.ipynb", "cpu-windows-2022-py3.11", "failed"),
        common.Member("c.ipynb", "dgpu-ubuntu-22.04-py3.11", "failed"),
    ]
    assert [m.key for m in evidence.pick_representatives(group, 2)] == [("a.ipynb", "cpu-ubuntu-22.04-py3.11"), ("b.ipynb", "cpu-windows-2022-py3.11")]
    assert [m.key for m in evidence.pick_representatives(group, 4)] == [
        ("a.ipynb", "cpu-ubuntu-22.04-py3.11"),
        ("b.ipynb", "cpu-windows-2022-py3.11"),
        ("c.ipynb", "dgpu-ubuntu-22.04-py3.11"),
        ("a.ipynb", "cpu-ubuntu-22.04-py3.12"),
    ]


def test_pick_representatives_of_a_single_notebook_spreads_over_configs():
    group = make_group("x")
    group.members = [common.Member("a.ipynb", f"cpu-ubuntu-22.04-py3.{v}", "failed") for v in (11, 12, 13)] + [
        common.Member("a.ipynb", "dgpu-ubuntu-22.04-py3.11", "failed")
    ]
    picked = evidence.pick_representatives(group, 3)
    assert len(picked) == 3 and len({m.config_id for m in picked}) == 3 and picked[0].config_id == "cpu-ubuntu-22.04-py3.11"


def test_pick_infra_jobs_prefers_distinct_configs():
    report = make_report(jobs=[make_job(1, CPU_CONFIG), make_job(2, CPU_CONFIG), make_job(3, GPU_CONFIG), make_job(4, CPU_CONFIG_312)])
    ctx = evidence.make_context(REPO, REPO, report, {})
    group = make_group("x", kind="infra")
    group.members = [common.Member(f"n{job_id}", "cfg", "not_run", job_id=job_id) for job_id in (1, 2, 3, 4, 99)]
    assert [j["id"] for j in evidence.pick_infra_jobs(group, ctx, 3)] == [1, 4, 3]


def test_job_log_digest_keeps_the_error_blocks_and_the_tail():
    lines = [f"2026-01-10T00:00:00.0000000Z \x1b[36mline {i}\x1b[0m" for i in range(1, 201)]
    lines[99] = "2026-01-10T00:00:00.0000000Z ##[error]Process completed with exit code 1."
    digest = evidence.job_log_digest("\r\n".join(lines))
    assert "--- lines 70-105 of the job log ---" in digest and "--- lines 121-200 of the job log ---" in digest
    assert "##[error]Process completed with exit code 1." in digest
    assert "line 70\n" in digest and "line 69\n" not in digest and "line 110\n" not in digest and "line 121" in digest
    assert "\x1b" not in digest and "2026-01-10T" not in digest


def test_job_log_digest_without_errors_is_the_tail_and_blocks_do_not_overlap():
    assert evidence.job_log_digest("a\nb").startswith("--- lines 1-2 of the job log ---")
    assert evidence.job_log_digest("\n".join(f"line {i}" for i in range(1, 300))).count("--- lines") == 1
    lines = [f"line {i}" for i in range(1, 101)]
    lines[49], lines[51] = "##[error] first", "##[error] second"
    body = [line for line in evidence.job_log_digest("\n".join(lines)).split("\n") if not line.startswith("---") and line]
    assert len(body) == len(set(body)) and "##[error] second" in body


def test_collect_job_logs_saves_extracts_and_tolerates_missing_logs(monkeypatch, tmp_path):
    def fake_gh(args, **kwargs):
        if args[1].endswith("/jobs/1/logs"):
            return b"2026-01-10T00:00:00.0000000Z ##[error]boom\n"
        raise common.GhError(args, 1, "HTTP 404: Not Found")

    monkeypatch.setattr(evidence, "gh", fake_gh)
    ctx = evidence.make_context(REPO, REPO, make_report(), {})
    warnings = []
    saved = evidence.collect_job_logs(ctx, [{"id": 1}, {"id": 2}], tmp_path / "logs", warnings.append)
    assert saved == {1: "logs/job-1-extract.log"}
    assert "##[error]boom" in (tmp_path / "logs" / "job-1-extract.log").read_text() and len(warnings) == 1


def test_relevant_packages_are_core_libraries_and_packages_named_in_the_error():
    env = {"openvino": "2026.1", "openvino-genai": "2026.1", "numpy": "2.0", "protobuf": "4.25", "Pillow": "11.0", "requests": "2.32"}
    picked = evidence.relevant_packages(env, "ImportError: cannot import name 'x' from 'google.protobuf' while loading pillow")
    assert set(picked) == {"openvino", "openvino-genai", "numpy", "protobuf", "Pillow"}
    assert len(evidence.relevant_packages(env, "", limit=2)) == 2
    assert evidence.normalize_package("Foo_Bar.baz") == "foo-bar-baz"


def test_environment_diff_lists_relevant_differences_first():
    failing = {"a": "1", "b": "2", "c": "3", "only-failing": "9"}
    passing = {"a": "1", "b": "3", "c": "3", "only-passing": "7"}
    assert evidence.environment_diff(failing, passing, {"only-passing"}) == ([("only-passing", "-", "7"), ("b", "2", "3"), ("only-failing", "9", "-")], 3)
    assert evidence.environment_diff(failing, passing, set(), limit=1) == ([("b", "2", "3")], 3)


def test_version_rows_compare_failing_and_passing_openvino_versions():
    report = make_report(
        results=[
            make_result("a/a.ipynb", "c1", "failed", env_id="e1", openvino="2026.1.0"),
            make_result("a/a.ipynb", "c2", "failed", env_id="e1", openvino="2026.1.0"),
            make_result("a/a.ipynb", "c3", "passed", env_id="e2", openvino="2026.0.0"),
            make_result("a/a.ipynb", "c4", "timeout"),
            make_result("other/other.ipynb", "c1", "passed", openvino="2026.1.0"),
        ]
    )
    group = make_group("x")
    group.members = [common.Member("a/a.ipynb", "c1", "failed"), common.Member("a/a.ipynb", "c2", "failed")]
    rows = evidence.version_rows(evidence.make_context(REPO, REPO, report, {}), group)
    assert rows == [
        {"openvino": "2026.1.0", "failed_with_error": 2, "passed": 0, "other": 0},
        {"openvino": "2026.0.0", "failed_with_error": 0, "passed": 1, "other": 0},
        {"openvino": "unknown", "failed_with_error": 0, "passed": 0, "other": 1},
    ]


def test_raw_logs_are_downloaded_once_per_fragment_and_stay_within_the_budget(monkeypatch, tmp_path):
    report = make_report(failures=[make_failure(f"n{i}/n.ipynb", CPU_CONFIG, SIG) for i in range(3)])
    downloads = []

    def fake_download(source_repo, artifact_id, dest):
        downloads.append(artifact_id)
        dest.write_bytes(make_zip({f["links"]["log_file"]: "x" * 100 for f in report["failures"]}))

    monkeypatch.setattr(evidence, "download_artifact", fake_download)
    ctx = evidence.make_context(REPO, REPO, report, {})
    members = [common.Member(f["notebook"], f["config_id"], "failed", failure=f) for f in report["failures"]]
    warnings = []
    saved = evidence.collect_raw_logs(ctx, members, tmp_path / "logs", tmp_path, [0], warnings.append)
    assert downloads == [77] and len(saved) == 3 and warnings == []

    monkeypatch.setattr(evidence, "MAX_TOTAL_LOG_BYTES", 250)
    saved = evidence.collect_raw_logs(ctx, members, tmp_path / "logs2", tmp_path, [0], warnings.append)
    assert len(saved) == 2 and "budget" in warnings[0]


# ----------------------------------------------------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------------------------------------------------


def test_fence_is_longer_than_any_backtick_run_in_the_text():
    assert render.fence("hello") == "```\nhello\n```"
    assert render.fence("\n\nhello\n", "python") == "```python\nhello\n```"
    assert render.fence("a\n```\nb").startswith("````\n") and render.fence("a\n```\nb").endswith("\n````")
    assert render.fence("`````x").startswith("``````\n")


def test_cell_clip_and_link_helpers():
    assert render.cell("a|b\n  c") == "a\\|b c" and render.cell("x" * 10, 5) == "xxxx…" and render.cell(None) == ""
    assert render.clip_tail("abcdef", 10) == "abcdef" and render.clip_tail("abcdef", 3) == "[3 earlier characters omitted]\ndef"
    assert render.clip_head("abcdef", 3) == "abc\n[3 more characters omitted]"
    assert render.link("x", "https://e.test") == "[x](https://e.test)" and render.link("x", None) == "x"
    assert render.repo_path("a/a.ipynb") == "notebooks/a/a.ipynb"


# ----------------------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------------------


def test_valid_args_pass():
    cnf.validate_args(make_args())
    cnf.validate_args(make_args(run_id="42540244557", source_repo="OpenVINOToolkit/openvino_notebooks"))
    cnf.validate_args(make_args(repo="akashchi/openvino_notebooks", source_repo="openvinotoolkit/openvino_notebooks"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"repo": "not a repo"},
        {"source_repo": "evil/repo"},
        {"run_id": "12; rm -rf /"},
        {"run_id": "1" * 21},
        {"max_groups": 0},
        {"max_groups": 11},
        {"max_prs": 0},
        {"history_nights": -1},
        {"history_nights": 8},
        {"logs_per_group": 0},
        {"logs_per_group": 7},
    ],
)
def test_invalid_args_are_rejected(overrides):
    with pytest.raises(SystemExit):
        cnf.validate_args(make_args(**overrides))


def test_parse_args_defaults_and_trimming():
    args = cnf.parse_args(["--repo", "a/b", "--out", "/tmp/nightly/digest", "--run-id", " 42 "])
    assert (args.source_repo, args.run_id, args.max_groups, args.max_prs, args.history_nights, args.logs_per_group) == ("a/b", "42", 5, 3, 3, 3)
    with pytest.raises(SystemExit):
        cnf.parse_args(["--repo", "a/b", "--source-repo", "evil/repo", "--out", "/tmp/nightly/digest"])


def test_prepare_out_dir_recreates_the_directory_and_refuses_risky_paths(tmp_path):
    target = tmp_path / "digest"
    target.mkdir()
    (target / "stale.txt").write_text("old")
    assert cnf.prepare_out_dir(str(target)) == target.resolve()
    assert list(target.iterdir()) == []
    for risky in ("/", "/tmp", str(Path.home()), str(Path.cwd())):
        with pytest.raises(SystemExit, match="refusing"):
            cnf.prepare_out_dir(risky)


def test_zip_member_names_are_never_extracted_to_disk(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "x")
    assert github.read_zip_member(archive, "../escape.txt", 100) == b"x"
    assert not (tmp_path.parent / "escape.txt").exists()
