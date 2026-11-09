import json
import re
from pathlib import Path

import pytest

import collector_analysis as analysis
import collector_common as common
import collector_github as github
import collector_evidence as evidence
import collect_nightly_failures as cnf
from nightly_builders import ENVIRONMENTS, NIGHTLY_ID, REPO, SIG_DEPENDENCY, SIG_OTHER, SIG_TIMEOUT, SIG_INFRA, FakeGh, make_report, nightly_world


@pytest.fixture(autouse=True)
def clear_github_output(monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def tracked_item(number, signature):
    return {
        "number": number,
        "title": f"Nightly remediation (sig {signature})",
        "body": f"Error signature: `{signature}`",
        "state": "OPEN",
        "url": f"https://github.com/{REPO}/pull/{number}",
        "createdAt": "2026-01-10T00:00:00Z",
        "closedAt": None,
        "mergedAt": None,
    }


def test_collect_writes_complete_digest_and_github_outputs(tmp_path, monkeypatch):
    fake = FakeGh()
    report = nightly_world(fake)
    out = tmp_path / "out" / "digest"
    output_file = tmp_path / "github-output"
    summary_file = tmp_path / "step-summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    monkeypatch.setattr(github, "gh", fake)
    monkeypatch.setattr(evidence, "gh", fake)

    assert cnf.main(["--repo", REPO, "--run-id", str(NIGHTLY_ID), "--out", str(out)]) == 0

    status = json.loads((out / "status.json").read_text())
    assert status["has_work"] is True
    assert status["failures_total"] == 4
    assert status["error_groups_total"] == 3
    assert status["infra_groups_total"] == 1
    assert status["infra_notebooks_not_run"] == 2
    assert status["groups_selected"] == 4
    assert "failure groups selected" in status["reason"]

    summary = (out / "summary.md").read_text()
    assert summary.index(SIG_DEPENDENCY) < summary.index(SIG_OTHER) < summary.index(SIG_INFRA) < summary.index(SIG_TIMEOUT)
    assert "PR candidate" in summary
    assert "usually an issue" in summary
    assert output_file.read_text() == "has_work=true\ngroups_selected=4\n"
    assert summary_file.read_text() == summary + "\n"

    group = (out / f"groups/01-{SIG_DEPENDENCY}/group.md").read_text()
    assert "Where it starts" in group
    assert "Failing cell source:" in group
    assert group.count("Failing cell source:") == 1
    assert group.count("Same notebook as an earlier failure above") == 1
    assert "Environment differences to a passing run" in group
    assert "`protobuf` | `4.25.0` | `6.31.0`" in group
    assert "| `2026.1.0` | 2 | 0 | 0 |" in group
    assert (out / f"groups/01-{SIG_DEPENDENCY}/group.json").exists()
    assert (out / f"groups/01-{SIG_DEPENDENCY}/logs/a-a.ipynb__cpu-ubuntu-22.04-py3.11.log").exists()

    raw_log = (out / f"groups/01-{SIG_DEPENDENCY}/logs/a-a.ipynb__cpu-ubuntu-22.04-py3.11.log").read_text()
    assert "\x1b" not in raw_log
    assert "\r" not in raw_log
    infra_log = (out / f"groups/03-{SIG_INFRA}/logs/job-13-extract.log").read_text()
    assert "##[error]Process completed with exit code 1." in infra_log
    assert "\x1b" not in infra_log
    assert not re.search(r"\d{4}-\d\d-\d\dT\d\d:\d\d:", infra_log)
    assert len(report["failures"]) == status["failures_total"]
    assert fake.count("--label nightly-remediator") == 2


def test_no_failures_is_successful_no_work_without_tracking_queries(tmp_path, monkeypatch):
    fake = FakeGh()
    fake.add_night(make_report(), 55)
    monkeypatch.setattr(github, "gh", fake)
    monkeypatch.setattr(evidence, "gh", fake)
    out = tmp_path / "out" / "digest"

    assert cnf.main(["--repo", REPO, "--run-id", str(NIGHTLY_ID), "--out", str(out)]) == 0

    status = json.loads((out / "status.json").read_text())
    assert status["has_work"] is False
    assert "no failures" in status["reason"]
    assert not (out / "groups").exists()
    assert fake.count("pr list") == 0


def test_open_items_block_all_groups_without_using_slots(tmp_path, monkeypatch):
    fake = FakeGh()
    nightly_world(fake)
    fake.prs = [tracked_item(index, signature) for index, signature in enumerate((SIG_DEPENDENCY, SIG_OTHER, SIG_INFRA, SIG_TIMEOUT), start=1)]
    monkeypatch.setattr(github, "gh", fake)
    monkeypatch.setattr(evidence, "gh", fake)
    out = tmp_path / "out" / "digest"

    assert cnf.main(["--repo", REPO, "--run-id", str(NIGHTLY_ID), "--out", str(out)]) == 0

    status = json.loads((out / "status.json").read_text())
    assert status["has_work"] is False
    assert status["groups_tracked"] == 4
    assert "already tracked" in status["reason"]
    assert "Already tracked" in (out / "summary.md").read_text()
    assert not (out / "groups").exists()


def test_disabled_issues_warns_but_open_pr_still_blocks(tmp_path, monkeypatch):
    fake = FakeGh()
    nightly_world(fake)
    fake.issues_disabled = True
    fake.prs = [tracked_item(1, SIG_DEPENDENCY)]
    monkeypatch.setattr(github, "gh", fake)
    monkeypatch.setattr(evidence, "gh", fake)
    out = tmp_path / "out" / "digest"

    assert cnf.main(["--repo", REPO, "--run-id", str(NIGHTLY_ID), "--out", str(out)]) == 0

    status = json.loads((out / "status.json").read_text())
    assert status["has_work"] is True
    assert any("issues are disabled" in warning for warning in status["warnings"])
    assert "Already tracked" in (out / "summary.md").read_text()
    assert not (out / f"groups/01-{SIG_DEPENDENCY}").exists()


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("unsupported schema", "unsupported nightly report schema"),
        ("in progress", "has not completed yet"),
        ("expired artifact", "not found"),
    ],
)
def test_expected_no_work_conditions_are_successful(tmp_path, monkeypatch, scenario, reason):
    fake = FakeGh()
    nightly_world(fake)
    if scenario == "unsupported schema":
        fake.add_night(make_report(schema_version="2.0.0"), 55, ENVIRONMENTS)
    elif scenario == "in progress":
        fake.runs[NIGHTLY_ID].update(status="in_progress")
    else:
        fake.artifacts[NIGHTLY_ID][0].update(expired=True)
    monkeypatch.setattr(github, "gh", fake)
    monkeypatch.setattr(evidence, "gh", fake)
    out = tmp_path / "out" / "digest"

    assert cnf.main(["--repo", REPO, "--run-id", str(NIGHTLY_ID), "--out", str(out)]) == 0

    status = json.loads((out / "status.json").read_text())
    assert status["has_work"] is False
    assert reason in status["reason"]


def test_wrong_workflow_and_api_failure_are_visible_errors(tmp_path, monkeypatch, capsys):
    fake = FakeGh()
    nightly_world(fake)
    fake.runs[NIGHTLY_ID]["path"] = ".github/workflows/other.yml"
    monkeypatch.setattr(github, "gh", fake)
    monkeypatch.setattr(evidence, "gh", fake)
    out = tmp_path / "wrong" / "digest"

    assert cnf.main(["--repo", REPO, "--run-id", str(NIGHTLY_ID), "--out", str(out)]) == 1
    assert "collector failed" in json.loads((out / "status.json").read_text())["reason"]
    assert "::error::" in capsys.readouterr().out

    fake = FakeGh()
    nightly_world(fake)
    fake.fail_on = "actions/runs/1001/artifacts?"
    monkeypatch.setattr(github, "gh", fake)
    monkeypatch.setattr(evidence, "gh", fake)
    out = tmp_path / "api-failure" / "digest"
    assert cnf.main(["--repo", REPO, "--run-id", str(NIGHTLY_ID), "--out", str(out)]) == 1
    assert "collector failed" in json.loads((out / "status.json").read_text())["reason"]
    assert "::error::" in capsys.readouterr().out


def test_automatic_resolution_skips_newer_cancelled_run(tmp_path, monkeypatch):
    fake = FakeGh()
    nightly_world(fake)
    fake.runs[1002] = {
        **fake.runs[NIGHTLY_ID],
        "id": 1002,
        "conclusion": "cancelled",
        "created_at": "2026-01-11T00:00:00Z",
    }
    monkeypatch.setattr(github, "gh", fake)
    monkeypatch.setattr(evidence, "gh", fake)
    out = tmp_path / "out" / "digest"

    assert cnf.main(["--repo", REPO, "--out", str(out)]) == 0
    assert json.loads((out / "status.json").read_text())["run_id"] == NIGHTLY_ID


def test_workflow_contract_stays_aligned_with_collector_cli_and_tracking():
    repo_root = Path(__file__).resolve().parents[4]
    workflow = (repo_root / ".github/workflows/nightly-issue-remediator.md").read_text()
    collector = (repo_root / ".github/scripts/agentic-workflows/collect_nightly_failures.py").read_text()

    assert "collect_nightly_failures.py" in workflow
    assert "TRACKING_LABEL" in workflow
    assert common.TRACKING_LABEL in workflow
    assert "Error signature: \\`<12-hex-signature>\\`" in workflow
    for flag in ("--repo", "--source-repo", "--run-id", "--max-groups", "--max-prs", "--out"):
        assert flag in workflow
        assert flag in collector


@pytest.mark.parametrize(
    "template,headings",
    [
        (
            "pr-template.md",
            ["Problem", "Error signature", "Affected notebooks", "Error type and where it starts", "Root cause", "Fix", "Verification and limits", "History"],
        ),
        (
            "issue-template.md",
            [
                "Summary",
                "Error signature",
                "Affected notebooks",
                "Error type and where it starts",
                "Evidence",
                "Investigation",
                "Why no automated fix",
                "Suggested next step",
            ],
        ),
    ],
)
def test_output_templates_are_imported_and_keep_tracking_contract(template, headings):
    repo_root = Path(__file__).resolve().parents[4]
    relative = f".github/workflows/shared/nightly-issue-remediator/{template}"
    workflow = (repo_root / ".github/workflows/nightly-issue-remediator.md").read_text()
    # The optional form keeps the template inline (the required form is hoisted to the top of the prompt), but silently skips a missing file.
    assert f"{{{{#runtime-import? {relative}}}}}" in workflow
    body = (repo_root / relative).read_text()

    assert re.findall(r"^## (.+)$", body, re.MULTILINE) == headings
    assert "Error signature: `<12-hex-signature>`\nNightly run: <nightly run URL>\n" in body
    assert "${{" not in body and "{{#" not in body
    filled = body.replace("<12-hex-signature>", SIG_DEPENDENCY)
    assert analysis.extract_signature(filled) == SIG_DEPENDENCY
