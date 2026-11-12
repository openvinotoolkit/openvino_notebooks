import copy
import json

import pytest

import knowledge_base as kb

SIG_A = "aaaaaaaaaaaa"
SIG_B = "bbbbbbbbbbbb"


def group(signature=SIG_A, **overrides):
    data = {
        "signature": signature,
        "kind": "error",
        "category": "dependency",
        "error_class": "dependency",
        "exception_type": "ModuleNotFoundError",
        "exception_message": "No module named 'x'",
        "notebooks": ["b/b.ipynb", "a/a.ipynb"],
    }
    data.update(overrides)
    return data


def attempt(run_id=1, date="2026-02-01", outcome="issue", **note):
    note.setdefault("outcome", outcome)
    return kb.make_attempt(run_id, f"https://github.com/o/r/actions/runs/{run_id}", "https://github.com/o/r/actions/runs/9", date, note)


def test_clean_text_bounds_and_flattens_untrusted_values():
    assert kb.clean_text("a\n\tb \x00c\u2028d") == "a b c d"
    assert kb.clean_text(None) == kb.clean_text(5) == ""
    assert kb.clean_text("x" * 1000) == "x" * 399 + "…"
    assert kb.clean_url("https://github.com/o/r/pull/1") == "https://github.com/o/r/pull/1"
    for bad in ("http://github.com/o", "https://evil.example/x", "https://github.com/a b", "https://github.com/" + "a" * 300, None):
        assert kb.clean_url(bad) == ""


def test_make_attempt_accepts_only_known_enums_and_bounds_every_note_field():
    result = attempt(outcome="merged", confidence="certain", root_cause="r" * 900, action="a\nb", extra="ignored")
    assert result["outcome"] == "none" and result["confidence"] == ""
    assert result["root_cause"] == "r" * 399 + "…" and result["action"] == "a b"
    assert result["dead_ends"] == result["next_step"] == ""
    assert set(result) == kb.ATTEMPT_KEYS
    assert kb.make_attempt(1, "x", "y", "2026-01-01", None)["outcome"] == "none"


def test_record_group_creates_then_updates_a_record():
    knowledge = kb.empty_knowledge()
    kb.record_group(knowledge, group(), attempt(1, "2026-02-01", "issue", root_cause="old"))
    record = knowledge["signatures"][SIG_A]
    assert record["investigations"] == 1 and record["first_seen"] == record["last_seen"] == "2026-02-01"
    assert record["notebooks"] == ["a/a.ipynb", "b/b.ipynb"]
    assert record["headline"] == "ModuleNotFoundError: No module named 'x'"

    kb.record_group(knowledge, group(notebooks=["c/c.ipynb"]), attempt(2, "2026-02-03", "pr", root_cause="new"))
    record = knowledge["signatures"][SIG_A]
    assert record["investigations"] == 2 and record["first_seen"] == "2026-02-01" and record["last_seen"] == "2026-02-03"
    assert [a["root_cause"] for a in record["attempts"]] == ["new", "old"]
    assert record["notebooks"] == ["c/c.ipynb"]
    kb.parse_knowledge(kb.dump_knowledge(knowledge))


def test_record_group_replaces_a_repeated_nightly_run_and_keeps_three_attempts():
    knowledge = kb.empty_knowledge()
    for run_id in range(1, 6):
        kb.record_group(knowledge, group(), attempt(run_id, f"2026-02-0{run_id}"))
    record = knowledge["signatures"][SIG_A]
    assert [a["nightly_run_id"] for a in record["attempts"]] == [5, 4, 3] and record["investigations"] == 5

    kb.record_group(knowledge, group(), attempt(4, "2026-02-04", "pr", root_cause="rerun"))
    record = knowledge["signatures"][SIG_A]
    assert record["investigations"] == 5
    assert [a["nightly_run_id"] for a in record["attempts"]] == [4, 5, 3] and record["attempts"][0]["root_cause"] == "rerun"
    assert record["last_seen"] == "2026-02-05"


def test_infra_headline_and_defaults_for_unexpected_group_values():
    knowledge = kb.empty_knowledge()
    infra = group(SIG_B, kind="infra", error_class="job_failed_before_tests", exception_type="", exception_message="Install step")
    kb.record_group(knowledge, infra, attempt())
    assert knowledge["signatures"][SIG_B]["headline"] == "job_failed_before_tests: Install step"
    kb.record_group(knowledge, group(kind="weird", category=None, exception_type=None, exception_message=None, notebooks=None), attempt())
    odd = knowledge["signatures"][SIG_A]
    assert odd["kind"] == "error" and odd["headline"] == "" and odd["notebooks"] == []


def test_eviction_drops_the_least_recently_seen_unprotected_signatures():
    knowledge = kb.empty_knowledge()
    for number in range(kb.MAX_SIGNATURES + 2):
        signature = f"{number:012x}"
        kb.record_group(knowledge, group(signature), attempt(number, f"2026-01-{number % 28 + 1:02d}"))
    protected = {f"{0:012x}"}
    before = {s: r["last_seen"] for s, r in knowledge["signatures"].items()}
    expected = sorted((s for s in before if s not in protected), key=lambda s: (before[s], s))[:2]
    evicted = kb.evict_oldest(knowledge, protected)
    assert evicted == expected
    assert len(knowledge["signatures"]) == kb.MAX_SIGNATURES
    assert f"{0:012x}" in knowledge["signatures"]


def test_eviction_enforces_the_size_cap_and_refuses_to_drop_protected_records(monkeypatch):
    knowledge = kb.empty_knowledge()
    for number in range(3):
        kb.record_group(knowledge, group(f"{number:012x}"), attempt(number, f"2026-01-0{number + 1}", root_cause="x" * 300))
    monkeypatch.setattr(kb, "MAX_FILE_BYTES", len(kb.dump_knowledge(knowledge)) - 1)
    assert kb.evict_oldest(copy.deepcopy(knowledge), set()) == [f"{0:012x}"]
    with pytest.raises(kb.KnowledgeError, match="alone exceed"):
        kb.evict_oldest(knowledge, set(knowledge["signatures"]))


def test_related_records_match_exception_type_or_notebook_most_recent_first():
    knowledge = kb.empty_knowledge()
    kb.record_group(knowledge, group("111111111111", notebooks=["z/z.ipynb"]), attempt(1, "2026-01-01"))
    kb.record_group(knowledge, group("222222222222", exception_type="OtherError", notebooks=["a/a.ipynb", "b/b.ipynb"]), attempt(2, "2026-01-05"))
    kb.record_group(knowledge, group("333333333333", exception_type="OtherError", notebooks=["q/q.ipynb"]), attempt(3, "2026-01-09"))
    kb.record_group(knowledge, group(SIG_A), attempt(4, "2026-01-10"))
    related = kb.related_records(knowledge, SIG_A, "ModuleNotFoundError", ["a/a.ipynb", "b/b.ipynb"])
    assert [(r["signature"], r["reasons"]) for r in related] == [
        ("222222222222", ["shares a/a.ipynb (+1 more)"]),
        ("111111111111", ["same exception type"]),
    ]
    assert kb.related_records(knowledge, SIG_A, "", [], limit=3) == []


def valid_knowledge():
    knowledge = kb.empty_knowledge()
    kb.record_group(knowledge, group(), attempt(root_cause="cause"))
    return knowledge


def invalid_documents():
    """Documents that every reader (Python here, the repo-memory validation script in the workflow) must reject."""
    text = kb.dump_knowledge(valid_knowledge())

    def mutate(change):
        data = json.loads(text)
        change(data)
        return json.dumps(data)

    record = lambda data: data["signatures"][SIG_A]  # noqa: E731
    return {
        "not json": "{",
        "top-level list": "[]",
        "version": mutate(lambda d: d.update(schema_version=2)),
        "bool version": mutate(lambda d: d.update(schema_version=True)),
        "extra top-level": mutate(lambda d: d.update(extra=1)),
        "bad key": mutate(lambda d: d["signatures"].update({"XYZ": d["signatures"].pop(SIG_A)})),
        "extra record field": mutate(lambda d: record(d).update(extra=1)),
        "missing record field": mutate(lambda d: record(d).pop("headline")),
        "bad kind": mutate(lambda d: record(d).update(kind="other")),
        "bad date": mutate(lambda d: record(d).update(first_seen="yesterday")),
        "zero investigations": mutate(lambda d: record(d).update(investigations=0)),
        "too many notebooks": mutate(lambda d: record(d).update(notebooks=["n"] * (kb.MAX_NOTEBOOKS + 1))),
        "no attempts": mutate(lambda d: record(d).update(attempts=[])),
        "too many attempts": mutate(lambda d: record(d).update(attempts=record(d)["attempts"] * (kb.MAX_ATTEMPTS + 1))),
        "bad outcome": mutate(lambda d: record(d)["attempts"][0].update(outcome="merged")),
        "bad confidence": mutate(lambda d: record(d)["attempts"][0].update(confidence="sure")),
        "bad url": mutate(lambda d: record(d)["attempts"][0].update(nightly_run_url="https://evil.example")),
        "long text": mutate(lambda d: record(d)["attempts"][0].update(root_cause="x" * (kb.TEXT_LIMIT + 1))),
        "long astral text": mutate(lambda d: record(d)["attempts"][0].update(root_cause="\U0001f600" * (kb.TEXT_LIMIT + 1))),
        "multi-line text": mutate(lambda d: record(d)["attempts"][0].update(action="a\nb")),
        "control character": mutate(lambda d: record(d)["attempts"][0].update(action="a\x85b")),
        "non-string text": mutate(lambda d: record(d)["attempts"][0].update(next_step=5)),
        "extra attempt field": mutate(lambda d: record(d)["attempts"][0].update(extra="x")),
        "bad run id": mutate(lambda d: record(d)["attempts"][0].update(nightly_run_id="1")),
        "too many signatures": json.dumps(
            {"schema_version": 1, "signatures": {f"{n:012x}": json.loads(text)["signatures"][SIG_A] for n in range(kb.MAX_SIGNATURES + 1)}}
        ),
    }


def boundary_documents():
    """Valid documents at the limits, which every reader must accept."""
    knowledge = valid_knowledge()
    record = knowledge["signatures"][SIG_A]
    record["attempts"][0]["root_cause"] = "\U0001f600" * kb.TEXT_LIMIT
    record["attempts"] = record["attempts"] * kb.MAX_ATTEMPTS
    record["notebooks"] = ["n"] * kb.MAX_NOTEBOOKS
    full = kb.empty_knowledge()
    for number in range(kb.MAX_SIGNATURES):
        full["signatures"][f"{number:012x}"] = record
    return {"empty": kb.dump_knowledge(kb.empty_knowledge()), "typical": kb.dump_knowledge(valid_knowledge()), "limits": kb.dump_knowledge(full)}


def test_parse_knowledge_roundtrips_and_rejects_every_schema_deviation():
    text = kb.dump_knowledge(valid_knowledge())
    assert kb.parse_knowledge(text) == json.loads(text)
    for name, value in boundary_documents().items():
        assert kb.parse_knowledge(value), name
    for name, value in invalid_documents().items():
        with pytest.raises(kb.KnowledgeError):
            kb.parse_knowledge(value)
            pytest.fail(f"accepted: {name}")
