import json

import pytest

import collector_analysis as analysis
import collector_feedback as feedback
import collector_github as github
import knowledge_base as kb
from collector_common import GhError, TrackedItem
from nightly_builders import FakeGh, REPO


def item(number, kind="pr"):
    return TrackedItem(
        kind,
        number,
        f"https://github.com/o/r/{'pull' if kind == 'pr' else 'issues'}/{number}",
        "Earlier remediation",
        "open",
        f"{number:012x}",
        f"2026-01-{number % 28 + 1:02d}",
    )


def response(kind="pr"):
    raw = {
        "title": "Earlier remediation",
        "url": "https://github.com/o/r/pull/42",
        "state": "OPEN",
        "body": "Evidence " * 600 + "\n## AI self-improvement\nA lesson at the end",
        "comments": {"nodes": [{"body": "Needs Windows verification", "url": "https://github.com/o/r/pull/42#issuecomment-1", "authorAssociation": "MEMBER"}]},
    }
    if kind == "pr":
        raw.update(
            baseRefOid="base-sha",
            headRefOid="head-sha",
            files={"nodes": [{"path": "requirements.txt"}], "pageInfo": {"hasNextPage": False}},
            reviews={
                "nodes": [
                    {
                        "state": "CHANGES_REQUESTED",
                        "body": "Does not fix the failing config",
                        "url": "https://github.com/o/r/pull/42#pullrequestreview-1",
                        "authorAssociation": "MEMBER",
                    }
                ]
            },
            reviewThreads={
                "nodes": [
                    {
                        "isResolved": False,
                        "isOutdated": True,
                        "path": "requirements.txt",
                        "comments": {
                            "nodes": [
                                {
                                    "body": "Pin still fails",
                                    "url": "https://github.com/o/r/pull/42#discussion_r1",
                                    "authorAssociation": "MEMBER",
                                    "commit": {"oid": "earlier-head"},
                                }
                            ]
                        },
                    }
                ]
            },
            commits={
                "nodes": [
                    {
                        "commit": {
                            "statusCheckRollup": {
                                "state": "FAILURE",
                                "contexts": {"nodes": [{"name": "Windows", "conclusion": "FAILURE"}], "pageInfo": {"hasNextPage": False}},
                            }
                        }
                    }
                ]
            },
        )
    return {"data": {"repository": {"pullRequest" if kind == "pr" else "issue": raw}}}


def test_feedback_is_bounded_deduplicated_and_keeps_observed_results_and_provenance(monkeypatch):
    tracked = {f"{number:012x}": [item(number)] for number in range(1, 10)}
    tracked["aaaaaaaaaaaa"] = [item(9)]  # One shared outcome must not consume two sample slots.
    tracked.update({f"{number:012x}": [item(number, "issue")] for number in range(10, 25)})
    calls = []

    def api(args):
        calls.append(args)
        query = next(arg for arg in args if arg.startswith("query="))
        assert "comments(last:3)" in query
        if "pullRequest(number:" in query:
            assert "reviews(last:3)" in query and "contexts(first:20)" in query and "files(first:8)" in query
        return response("pr" if "pullRequest(number:" in query else "issue")

    monkeypatch.setattr(github, "list_json", api)
    result = feedback.fetch_feedback("o/r", tracked, kb.empty_knowledge(), lambda message: None)
    assert len(calls) == len(result["items"]) == 12
    assert [record["number"] for record in result["items"][:4]] == [9, 8, 7, 6]
    record = result["items"][0]
    assert record["signatures"] == ["000000000009", "aaaaaaaaaaaa"]
    assert record["head_sha"] == "head-sha" and record["base_sha"] == "base-sha"
    assert record["checks"][0]["state"] == "FAILURE"
    assert record["reviews"][0]["state"] == "CHANGES_REQUESTED"
    assert record["comments"][0]["authorAssociation"] == "MEMBER"
    assert record["review_threads"][0]["isOutdated"]
    assert record["review_threads"][0]["comments"][0]["commit"]["oid"] == "earlier-head"
    assert record["self_improvement"] == "A lesson at the end" and record["body_truncated"]
    assert len(record["body_excerpt"]) <= feedback.BODY_LIMIT
    assert "head_sha" not in result["items"][-1]
    assert "not proof of correctness" in feedback.render_feedback(result)


def test_feedback_errors_are_explicit_and_do_not_erase_other_evidence(monkeypatch):
    calls = []
    warnings = []

    def api(args):
        calls.append(args)
        if len(calls) == 1:
            raise GhError(args, 1, "HTTP 403: unavailable")
        return {"errors": [{"message": "no access"}]}

    monkeypatch.setattr(github, "list_json", api)
    prior = item(1)
    prior.self_improvement = "For missing imports after pip failure, inspect installation logs before proposing a pin."
    result = feedback.fetch_feedback("o/r", {"a": [prior, item(2)]}, kb.empty_knowledge(), warnings.append)
    assert not result["items"] and len(result["unavailable"]) == len(warnings) == 2
    assert all("unavailable" in warning for warning in warnings)
    assert result["self_improvements"][0]["self_improvement"] == prior.self_improvement


def test_recent_learning_from_other_signatures_is_exposed_without_treating_it_as_verified(monkeypatch):
    knowledge = kb.empty_knowledge()
    for number in range(15):
        group = {"signature": f"{number:012x}", "notebooks": ["notebooks/other/other.ipynb"]}
        attempt = kb.make_attempt(
            number,
            "",
            f"https://github.com/o/r/actions/runs/{number}",
            "2026-01-01",
            {"self_assessment": "requested, not validated", "lesson": f"lesson {number}"},
        )
        kb.record_group(knowledge, group, attempt)
    result = feedback.fetch_feedback("o/r", {}, knowledge, lambda message: None)
    assert len(result["lessons"]) == 10 and result["lessons"][0]["lesson"] == "lesson 14"
    assert result["lessons"][0]["remediator_run_url"].endswith("/14")
    assert all(lesson["self_assessment"] == "requested, not validated" for lesson in result["lessons"])
    assert json.loads(json.dumps(result)) == result


def test_new_feedback_on_old_items_is_prioritized_and_repeated_lessons_do_not_fill_the_sample(monkeypatch):
    old = item(1)
    old.updated_at = "2026-03-01"
    tracked = {f"{number:012x}": [item(number)] for number in range(2, 8)}
    tracked[old.signature] = [old]
    monkeypatch.setattr(github, "list_json", lambda args: response())
    knowledge = kb.empty_knowledge()
    for number in range(15):
        kb.record_group(knowledge, {"signature": f"{number:012x}"}, kb.make_attempt(number, "", "", "2026-01-01", {"lesson": "same lesson"}))
    result = feedback.fetch_feedback("o/r", tracked, knowledge, lambda message: None)
    assert result["items"][0]["number"] == 1
    assert len(result["lessons"]) == 1
    assert result["lessons"][0]["signature"] == "00000000000e"


def test_large_affected_tables_do_not_hide_causal_evidence_from_the_feedback_excerpt(monkeypatch):
    data = response()
    data["data"]["repository"]["pullRequest"]["body"] = (
        "## Affected notebooks\n" + "| notebook | config |\n" * 1000 + "\n## Root cause\nA stale transitive dependency\n## Fix\nRaise the verified minimum\n"
        "## Verification and limits\nWindows remains unverified\n"
    )
    monkeypatch.setattr(github, "list_json", lambda args: data)
    record = feedback.fetch_feedback("o/r", {"000000000001": [item(1)]}, kb.empty_knowledge(), lambda message: None)["items"][0]
    assert "A stale transitive dependency" in record["body_excerpt"]
    assert "Raise the verified minimum" in record["body_excerpt"]
    assert "Windows remains unverified" in record["body_excerpt"]
    assert record["body_truncated"]


@pytest.mark.parametrize(
    "body",
    [
        "## AI self-improvement\nCheck the earlier install failure first.\n## Other evidence\nNot a lesson.",
        "<details>\n<summary>AI self-improvement</summary>\n\nCheck the earlier install failure first.\n</details>\n## Other evidence\nNot a lesson.",
    ],
)
def test_learning_extraction_supports_legacy_and_collapsed_sections_without_surrounding_text(body):
    assert feedback.body_section(body, "AI self-improvement", 2000) == "Check the earlier install failure first."


def test_preflight_collects_learning_from_each_tracked_outcome_not_only_the_rich_feedback_sample(monkeypatch):
    fake = FakeGh()
    for kind, count in (("pr", 6), ("issue", 10)):
        for number in range(1, count + 1):
            signature = f"{number:012x}"
            raw = {
                "number": number,
                "title": f"Earlier remediation (sig {signature})",
                "body": (
                    f"Error signature: `{signature}`\nRelated signatures: `aaaaaaaaaaaa`\n"
                    "<details>\n<summary>AI self-improvement</summary>\n\n"
                    f"Future improvement from {kind} {number}: trace the install failure first.\n</details>"
                ),
                "state": "OPEN",
                "url": f"https://github.com/{REPO}/{'pull' if kind == 'pr' else 'issues'}/{number}",
                "createdAt": f"2026-01-{number:02d}T00:00:00Z",
            }
            (fake.prs if kind == "pr" else fake.issues).append(raw)
    monkeypatch.setattr(github, "gh", fake)

    tracked = analysis.fetch_tracked(REPO, lambda message: None)
    result = feedback.fetch_feedback(REPO, tracked, kb.empty_knowledge(), lambda message: None)

    assert len(result["items"]) == 12
    assert len(result["self_improvements"]) == 16
    lessons = {(record["kind"], record["number"]): record for record in result["self_improvements"]}
    assert lessons[("pr", 1)]["self_improvement"] == "Future improvement from pr 1: trace the install failure first."
    assert lessons[("issue", 1)]["self_improvement"] == "Future improvement from issue 1: trace the install failure first."
    assert lessons[("issue", 1)]["url"].endswith("/issues/1")
    assert lessons[("pr", 1)]["signatures"] == ["000000000001", "aaaaaaaaaaaa"]
    assert not lessons[("pr", 1)]["self_improvement_truncated"]
    assert ("pr", 1) not in {(record["kind"], record["number"]) for record in result["items"]}
    assert fake.count("api graphql") == 16, "learning extraction must reuse tracking bodies, not add per-item API calls"


def test_learning_collection_reports_missing_sections_and_truncation(monkeypatch):
    fake = FakeGh()
    fake.issues = [
        {
            "number": number,
            "title": f"Earlier remediation (sig {number:012x})",
            "body": "## AI self-improvement\n" + "lesson " * 1000 if number == 1 else "Legacy item without learning",
            "state": "OPEN",
            "url": f"https://github.com/{REPO}/issues/{number}",
            "createdAt": "2026-01-01T00:00:00Z",
        }
        for number in (1, 2)
    ]
    monkeypatch.setattr(github, "gh", fake)
    tracked = analysis.fetch_tracked(REPO, lambda message: None)
    result = feedback.fetch_feedback(REPO, tracked, kb.empty_knowledge(), lambda message: None)
    lessons = {record["number"]: record for record in result["self_improvements"]}

    assert lessons[1]["self_improvement_truncated"]
    assert len(lessons[1]["self_improvement"]) == feedback.SELF_IMPROVEMENT_LIMIT
    assert lessons[2]["self_improvement"] == ""
    assert not lessons[2]["self_improvement_truncated"]
