import json
import shutil
from pathlib import Path

import pytest

import knowledge_base as kb
import update_knowledge_base as ukb

SIG_A = "aaaaaaaaaaaa"
SIG_B = "bbbbbbbbbbbb"
RUN_URL = "https://github.com/o/r/actions/runs/1001"
REMEDIATOR_URL = "https://github.com/o/r/actions/runs/2002"


def group_json(index, signature, **overrides):
    data = {
        "index": index,
        "signature": signature,
        "kind": "error",
        "category": "dependency",
        "error_class": "dependency",
        "exception_type": "ModuleNotFoundError",
        "exception_message": "No module named 'x'",
        "notebooks": ["a/a.ipynb"],
        "raw_logs": {"a/a.ipynb::cpu": "logs/a.log"},
    }
    data.update(overrides)
    return data


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_digest(root, groups=None, has_work=True, run=None):
    groups = [group_json(1, SIG_A), group_json(2, SIG_B, exception_type="OtherError")] if groups is None else groups
    write_json(root / "status.json", {"has_work": has_work})
    write_json(root / "run.json", run or {"id": 1001, "url": RUN_URL, "nightly_date": "2026-02-01"})
    for item in groups:
        write_json(root / "groups" / f"{item['index']:02d}-{item['signature']}" / "group.json", item)
        (root / "groups" / f"{item['index']:02d}-{item['signature']}" / "group.md").write_text("big markdown", encoding="utf-8")
        (root / "groups" / f"{item['index']:02d}-{item['signature']}" / "logs").mkdir(exist_ok=True)
    return root


class World:
    def __init__(self, tmp_path):
        self.digest = make_digest(tmp_path / "digest")
        self.memory = tmp_path / "memory"
        self.memory.mkdir()
        self.trusted = tmp_path / "trusted"
        self.notes = self.digest / "outcomes"
        self.warnings = []

    def snapshot(self):
        ukb.snapshot(self.digest, self.memory, self.trusted, self.notes)

    def note(self, signature, **fields):
        write_json(self.notes / f"{signature}.json", fields)

    def update(self):
        return ukb.update(self.trusted, self.notes, self.memory, REMEDIATOR_URL, "2026-03-01", warn=self.warnings.append)

    def stored(self):
        return kb.parse_knowledge((self.memory / kb.KNOWLEDGE_FILE).read_text(encoding="utf-8"))


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_snapshot_copies_only_what_the_update_needs(world):
    (world.memory / kb.KNOWLEDGE_FILE).write_text(kb.dump_knowledge(kb.empty_knowledge()), encoding="utf-8")
    world.snapshot()
    copied = sorted(p.relative_to(world.trusted).as_posix() for p in world.trusted.rglob("*") if p.is_file())
    assert copied == [
        "digest/groups/01-aaaaaaaaaaaa/group.json",
        "digest/groups/02-bbbbbbbbbbbb/group.json",
        "digest/run.json",
        "digest/status.json",
        "memory/knowledge.json",
    ]
    assert world.notes.is_dir()


def test_snapshot_without_a_baseline_and_with_a_stale_snapshot(world):
    world.trusted.mkdir()
    (world.trusted / "stale.txt").write_text("old", encoding="utf-8")
    world.snapshot()
    assert not (world.trusted / "stale.txt").exists() and not (world.trusted / "memory").exists()
    (world.digest / "status.json").unlink()
    with pytest.raises(kb.KnowledgeError):
        world.snapshot()


def test_update_records_every_investigated_group_with_the_agents_notes(world):
    world.snapshot()
    world.note(
        SIG_A, outcome="pr", confidence="high", root_cause="pinned protobuf", action="pin protobuf<5", dead_ends="none", next_step="watch the next nightly"
    )
    world.note(SIG_B, outcome="draft", confidence="certain", root_cause="r" * 900)
    result = world.update()
    assert result == {"investigated": 2, "evicted": [], "signatures": 2}

    stored = world.stored()
    record = stored["signatures"][SIG_A]
    assert record["investigations"] == 1 and record["first_seen"] == record["last_seen"] == "2026-02-01"
    assert record["attempts"][0] == {
        "nightly_run_id": 1001,
        "nightly_run_url": RUN_URL,
        "remediator_run_url": REMEDIATOR_URL,
        "date": "2026-02-01",
        "outcome": "pr",
        "confidence": "high",
        "root_cause": "pinned protobuf",
        "action": "pin protobuf<5",
        "dead_ends": "none",
        "next_step": "watch the next nightly",
    }
    other = stored["signatures"][SIG_B]["attempts"][0]
    assert other["outcome"] == "none" and other["confidence"] == "" and other["root_cause"] == "r" * 399 + "…"


def test_update_without_notes_still_counts_the_investigation(world):
    world.snapshot()
    world.update()
    attempt = world.stored()["signatures"][SIG_A]["attempts"][0]
    assert attempt["outcome"] == "none" and attempt["root_cause"] == "" and world.warnings == []


def test_update_builds_on_the_baseline_and_keeps_other_signatures(world):
    baseline = kb.empty_knowledge()
    other = "cccccccccccc"
    kb.record_group(baseline, group_json(1, other), kb.make_attempt(900, RUN_URL, REMEDIATOR_URL, "2026-01-01", {"outcome": "issue", "root_cause": "kept"}))
    kb.record_group(baseline, group_json(1, SIG_A), kb.make_attempt(900, RUN_URL, REMEDIATOR_URL, "2026-01-01", {"outcome": "issue", "root_cause": "earlier"}))
    (world.memory / kb.KNOWLEDGE_FILE).write_text(kb.dump_knowledge(baseline), encoding="utf-8")
    world.snapshot()
    world.update()

    stored = world.stored()
    assert stored["signatures"][other] == baseline["signatures"][other]
    record = stored["signatures"][SIG_A]
    assert record["investigations"] == 2 and record["first_seen"] == "2026-01-01" and record["last_seen"] == "2026-02-01"
    assert [a["root_cause"] for a in record["attempts"]] == ["", "earlier"]


def test_update_ignores_everything_the_agent_changes_after_the_snapshot(world):
    baseline = kb.empty_knowledge()
    kb.record_group(baseline, group_json(1, SIG_A), kb.make_attempt(900, RUN_URL, REMEDIATOR_URL, "2026-01-01", {"root_cause": "trusted"}))
    (world.memory / kb.KNOWLEDGE_FILE).write_text(kb.dump_knowledge(baseline), encoding="utf-8")
    world.snapshot()

    # The agent edits the persisted file, forges the digest and plants another signature.
    (world.memory / kb.KNOWLEDGE_FILE).write_text('{"forged": true}', encoding="utf-8")
    shutil.rmtree(world.digest / "groups" / f"01-{SIG_A}")
    write_json(world.digest / "groups" / "09-dddddddddddd" / "group.json", group_json(9, "dddddddddddd", exception_message="injected"))
    write_json(world.digest / "status.json", {"has_work": False})
    world.note("dddddddddddd", root_cause="planted")
    world.note(SIG_A, root_cause="honest note")

    world.update()
    stored = world.stored()
    assert set(stored["signatures"]) == {SIG_A, SIG_B}
    assert [a["root_cause"] for a in stored["signatures"][SIG_A]["attempts"]] == ["honest note", "trusted"]
    assert any("dddddddddddd.json" in warning for warning in world.warnings)


def test_update_does_nothing_when_no_group_was_selected(tmp_path):
    world = World(tmp_path)
    make_digest(world.digest, groups=[], has_work=False)
    world.snapshot()
    assert world.update() == {"investigated": 0, "evicted": [], "signatures": None}
    assert not (world.memory / kb.KNOWLEDGE_FILE).exists()


@pytest.mark.parametrize("content", ["{", "[]", json.dumps({"root_cause": "x" * ukb.MAX_NOTE_BYTES})])
def test_malformed_notes_are_ignored_with_a_warning(world, content):
    world.snapshot()
    (world.notes / f"{SIG_A}.json").write_text(content, encoding="utf-8")
    world.update()
    assert any(f"{SIG_A}.json" in warning for warning in world.warnings)
    assert world.stored()["signatures"][SIG_A]["attempts"][0]["root_cause"] == ""


@pytest.mark.parametrize("name", ["notes.txt", "zzzzzzzzzzzz.json", "cccccccccccc.json", "AAAAAAAAAAAA.json", "../escape.json"])
def test_notes_for_unexpected_names_or_unselected_signatures_are_ignored(world, name):
    world.snapshot()
    (world.notes / Path(name).name).write_text('{"root_cause": "planted"}', encoding="utf-8")
    world.update()
    assert any(Path(name).name in warning for warning in world.warnings)
    assert set(world.stored()["signatures"]) == {SIG_A, SIG_B}
    assert all(a["root_cause"] == "" for r in world.stored()["signatures"].values() for a in r["attempts"])


def test_a_directory_named_like_a_note_is_ignored(world):
    world.snapshot()
    (world.notes / f"{SIG_A}.json").mkdir()
    world.update()
    assert any(f"{SIG_A}.json" in warning for warning in world.warnings)


def test_corrupt_baseline_fails_without_touching_the_memory_directory(world):
    (world.memory / kb.KNOWLEDGE_FILE).write_text("{", encoding="utf-8")
    world.snapshot()
    with pytest.raises(kb.KnowledgeError):
        world.update()
    assert (world.memory / kb.KNOWLEDGE_FILE).read_text(encoding="utf-8") == "{"


def test_update_evicts_the_least_recently_seen_signatures_beyond_the_cap(world, monkeypatch):
    baseline = kb.empty_knowledge()
    for number in range(kb.MAX_SIGNATURES):
        kb.record_group(baseline, group_json(1, f"{number:012x}"), kb.make_attempt(1, RUN_URL, REMEDIATOR_URL, f"2026-01-{number % 28 + 1:02d}", None))
    (world.memory / kb.KNOWLEDGE_FILE).write_text(kb.dump_knowledge(baseline), encoding="utf-8")
    world.snapshot()
    result = world.update()
    assert len(world.stored()["signatures"]) == kb.MAX_SIGNATURES and len(result["evicted"]) == 2
    assert {SIG_A, SIG_B} <= set(world.stored()["signatures"])


@pytest.mark.parametrize(
    "run,expected",
    [
        ({"id": 1, "nightly_date": "2026-02-02", "created_at": "2026-02-03T00:00:00Z"}, "2026-02-02"),
        ({"id": 1, "nightly_date": "garbage", "created_at": "2026-02-03T00:00:00Z"}, "2026-02-03"),
        ({"id": 1}, "2026-03-01"),
    ],
)
def test_nightly_date_falls_back_to_the_creation_date_then_today(world, run, expected):
    write_json(world.digest / "run.json", run)
    world.snapshot()
    world.update()
    assert world.stored()["signatures"][SIG_A]["last_seen"] == expected


def test_update_requires_a_numeric_run_id(world):
    write_json(world.digest / "run.json", {"id": "1001"})
    world.snapshot()
    with pytest.raises(kb.KnowledgeError, match="run id"):
        world.update()


def cli(world, command):
    common = ["--trusted-dir", str(world.trusted), "--notes-dir", str(world.notes), "--memory-dir", str(world.memory)]
    if command == "snapshot":
        return ukb.main(["snapshot", "--digest-dir", str(world.digest), *common])
    return ukb.main(["update", "--remediator-run-url", REMEDIATOR_URL, *common])


def test_cli_snapshot_then_update(world, capsys):
    assert cli(world, "snapshot") == 0
    world.note(SIG_A, outcome="issue", root_cause="from the cli")
    assert cli(world, "update") == 0
    assert "Recorded 2 investigated signatures" in capsys.readouterr().out
    assert world.stored()["signatures"][SIG_A]["attempts"][0]["root_cause"] == "from the cli"


def test_cli_reports_failures_as_errors_and_no_work_as_unchanged(world, capsys):
    (world.memory / kb.KNOWLEDGE_FILE).write_text("{", encoding="utf-8")
    assert cli(world, "snapshot") == 0
    assert cli(world, "update") == 1
    assert "::error::knowledge base update failed" in capsys.readouterr().out

    write_json(world.digest / "status.json", {"has_work": False})
    (world.memory / kb.KNOWLEDGE_FILE).unlink()
    assert cli(world, "snapshot") == 0 and cli(world, "update") == 0
    assert "knowledge base is unchanged" in capsys.readouterr().out
    assert Path(world.memory / kb.KNOWLEDGE_FILE).exists() is False
