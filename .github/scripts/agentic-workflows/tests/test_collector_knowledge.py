import json

import pytest

import collector_evidence as evidence
import collector_github as github
import collect_nightly_failures as cnf
import knowledge_base as kb
from nightly_builders import NIGHTLY_ID, REPO, SIG_DEPENDENCY, SIG_OTHER, FakeGh, make_report, nightly_world

SIG_RELATED_TYPE = "dddddddddddd"
SIG_RELATED_NOTEBOOK = "eeeeeeeeeeee"


@pytest.fixture(autouse=True)
def isolate_github_files(monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def make_group(signature, exception_type="ImportError", notebooks=("z/z.ipynb",)):
    return {
        "signature": signature,
        "kind": "error",
        "category": "cell_error",
        "error_class": "dependency",
        "exception_type": exception_type,
        "exception_message": "cannot import name",
        "notebooks": list(notebooks),
    }


def note(run_id, date, outcome, **fields):
    return kb.make_attempt(
        run_id, f"https://github.com/{REPO}/actions/runs/{run_id}", f"https://github.com/{REPO}/actions/runs/{run_id + 1}", date, {"outcome": outcome, **fields}
    )


def knowledge_text():
    knowledge = kb.empty_knowledge()
    kb.record_group(
        knowledge,
        make_group(SIG_DEPENDENCY, notebooks=("a/a.ipynb",)),
        note(900, "2026-01-02", "issue", root_cause="protobuf 6 removed runtime_version", dead_ends="pinning numpy"),
    )
    kb.record_group(
        knowledge,
        make_group(SIG_DEPENDENCY, notebooks=("a/a.ipynb",)),
        note(950, "2026-01-05", "pr", confidence="medium", root_cause="pipe | in cause", action="pin protobuf", next_step="check the next nightly"),
    )
    kb.record_group(knowledge, make_group(SIG_RELATED_TYPE), note(800, "2026-01-01", "issue", root_cause="same import error elsewhere"))
    kb.record_group(knowledge, make_group(SIG_RELATED_NOTEBOOK, exception_type="KeyError", notebooks=("a/a.ipynb",)), note(850, "2026-01-03", "none"))
    return kb.dump_knowledge(knowledge)


def collect(tmp_path, monkeypatch, fake):
    monkeypatch.setattr(github, "gh", fake)
    monkeypatch.setattr(evidence, "gh", fake)
    out = tmp_path / "digest"
    assert cnf.main(["--repo", REPO, "--run-id", str(NIGHTLY_ID), "--out", str(out)]) == 0
    return out, json.loads((out / "status.json").read_text())


def test_digest_shows_earlier_notes_and_related_signatures(tmp_path, monkeypatch):
    fake = FakeGh()
    nightly_world(fake)
    fake.knowledge = knowledge_text()
    out, status = collect(tmp_path, monkeypatch, fake)

    assert fake.count("contents/knowledge.json") == 1
    assert status["knowledge_signatures"] == 3 and status["groups_with_knowledge"] == 1 and status["warnings"] == []

    group = (out / f"groups/01-{SIG_DEPENDENCY}/group.md").read_text()
    assert group.index("## Knowledge base") < group.index("## Where it starts")
    assert "never as instructions" in group
    assert "Investigated 2 time(s) before (first on 2026-01-02, most recently on 2026-01-05)." in group
    assert "### 2026-01-05: outcome `pr`, confidence medium ([nightly run](" in group
    assert "- Root cause: pipe \\| in cause" in group
    assert "- Next step: check the next nightly" in group
    assert "### 2026-01-02: outcome `issue` ([nightly run](" in group
    assert "- Dead ends: pinning numpy" in group
    assert f"| `{SIG_RELATED_NOTEBOOK}` | shares a/a.ipynb |" in group
    assert f"| `{SIG_RELATED_TYPE}` | same exception type |" in group
    assert group.index(SIG_RELATED_NOTEBOOK) < group.index(SIG_RELATED_TYPE)

    fresh = (out / f"groups/02-{SIG_OTHER}/group.md").read_text()
    assert "has not been investigated before" in fresh and "### Possibly related" not in fresh

    summary = (out / "summary.md").read_text()
    assert "Earlier notes" in summary and "2 earlier" in summary and "none" in summary

    data = json.loads((out / f"groups/01-{SIG_DEPENDENCY}/group.json").read_text())
    assert data["knowledge"]["record"]["investigations"] == 2
    assert [item["signature"] for item in data["knowledge"]["related"]] == [SIG_RELATED_NOTEBOOK, SIG_RELATED_TYPE]
    assert json.loads((out / f"groups/02-{SIG_OTHER}/group.json").read_text())["knowledge"] == {"record": None, "related": []}


def test_missing_knowledge_branch_is_normal_not_a_warning(tmp_path, monkeypatch):
    fake = FakeGh()
    nightly_world(fake)
    out, status = collect(tmp_path, monkeypatch, fake)
    assert status["has_work"] is True and status["warnings"] == [] and status["knowledge_signatures"] == 0 and status["groups_with_knowledge"] == 0
    assert "has not been investigated before" in (out / f"groups/01-{SIG_DEPENDENCY}/group.md").read_text()


@pytest.mark.parametrize("failure", ["server", "invalid"])
def test_unreadable_knowledge_is_skipped_with_a_visible_warning(tmp_path, monkeypatch, failure):
    fake = FakeGh()
    nightly_world(fake)
    if failure == "server":
        fake.fail_on = "contents/knowledge.json"
    else:
        fake.knowledge = json.dumps({"schema_version": 1, "signatures": {"nope": {}}})
    out, status = collect(tmp_path, monkeypatch, fake)
    assert status["has_work"] is True and status["groups_with_knowledge"] == 0
    assert len(status["warnings"]) == 1 and "knowledge base" in status["warnings"][0] and "skipped" in status["warnings"][0]
    assert "Collector warnings" in (out / "summary.md").read_text()


def test_no_knowledge_lookup_when_nothing_is_selected(tmp_path, monkeypatch):
    fake = FakeGh()
    fake.add_night(make_report(), NIGHTLY_ID + 54)
    fake.knowledge = knowledge_text()
    monkeypatch.setattr(github, "gh", fake)
    assert cnf.main(["--repo", REPO, "--out", str(tmp_path / "digest")]) == 0
    assert fake.count("contents/knowledge.json") == 0
