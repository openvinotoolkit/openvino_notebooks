"""Records what the nightly remediator learned into the knowledge base kept on the `repo-memory` branch.

`snapshot` runs before the agent and freezes what the update trusts outside the agent's sandbox: the digest, the knowledge base as cloned, and
the updater scripts themselves (copied last, so their presence marks a complete snapshot).
`update` runs after the agent from that script copy. It rebuilds `knowledge.json` into a separate output directory from the snapshots plus the
agent's per-signature notes (`outcomes/<signature>.json`), so the agent never writes the persisted file itself and cannot change counts, dates,
or other signatures' records.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from knowledge_base import (
    DATE_RE,
    KNOWLEDGE_FILE,
    SIGNATURE_KEY_RE,
    KnowledgeError,
    clean_text,
    dump_knowledge,
    empty_knowledge,
    evict_oldest,
    make_attempt,
    parse_knowledge,
    record_group,
)

MAX_NOTE_BYTES = 16 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
DIGEST_SNAPSHOT = "digest"
MEMORY_SNAPSHOT = "memory"
SCRIPTS_SNAPSHOT = "scripts"
SCRIPT_FILES = ("knowledge_base.py", "update_knowledge_base.py")


def read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise KnowledgeError(f"{path} is missing or too large")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise KnowledgeError(f"{path} does not contain a JSON object")
    return data


def snapshot(digest_dir: Path, memory_dir: Path, trusted_dir: Path, notes_dir: Path) -> None:
    shutil.rmtree(trusted_dir, ignore_errors=True)
    digest_copy = trusted_dir / DIGEST_SNAPSHOT
    digest_copy.mkdir(parents=True)
    read_json(digest_dir / "status.json")
    for name in ("status.json", "run.json"):
        if (digest_dir / name).is_file():
            shutil.copy2(digest_dir / name, digest_copy / name)
    for group_json in sorted((digest_dir / "groups").glob("*/group.json")):
        target = digest_copy / "groups" / group_json.parent.name
        target.mkdir(parents=True)
        shutil.copy2(group_json, target / group_json.name)
    baseline = memory_dir / KNOWLEDGE_FILE
    if baseline.is_file():
        (trusted_dir / MEMORY_SNAPSHOT).mkdir()
        shutil.copy2(baseline, trusted_dir / MEMORY_SNAPSHOT / KNOWLEDGE_FILE)
    notes_dir.mkdir(parents=True, exist_ok=True)
    scripts = trusted_dir / SCRIPTS_SNAPSHOT
    scripts.mkdir()
    for name in SCRIPT_FILES:
        shutil.copy2(Path(__file__).resolve().parent / name, scripts / name)


def read_notes(notes_dir: Path, signatures: List[str], warn=print) -> Dict[str, dict]:
    """The agent's notes for the investigated signatures; anything malformed or unexpected is ignored."""
    notes: Dict[str, dict] = {}
    if not notes_dir.is_dir():
        return notes
    for path in sorted(notes_dir.iterdir()):
        signature = path.stem
        if path.suffix != ".json" or not SIGNATURE_KEY_RE.match(signature) or signature not in signatures:
            warn(f"ignoring unexpected knowledge note {path.name}")
            continue
        try:
            if not path.is_file() or path.stat().st_size > MAX_NOTE_BYTES:
                raise ValueError("not a regular file or too large")
            note = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(note, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as error:
            warn(f"ignoring knowledge note {path.name}: {error}")
            continue
        notes[signature] = note
    return notes


def nightly_date(run: dict, today: str) -> str:
    for value in (run.get("nightly_date"), str(run.get("created_at") or "")[:10]):
        if isinstance(value, str) and DATE_RE.match(value):
            return value
    return today


def update(trusted_dir: Path, notes_dir: Path, out_dir: Path, remediator_run_url: str, today: str, warn=print) -> dict:
    digest = trusted_dir / DIGEST_SNAPSHOT
    if not read_json(digest / "status.json").get("has_work"):
        return {"investigated": 0, "evicted": [], "signatures": None}
    run = read_json(digest / "run.json")
    run_id = run.get("id")
    if type(run_id) is not int:
        raise KnowledgeError("run.json has no numeric run id")
    date = nightly_date(run, today)
    groups = sorted((read_json(path) for path in (digest / "groups").glob("*/group.json")), key=lambda g: g.get("index", 0))
    signatures = [s for s in (g.get("signature") for g in groups) if isinstance(s, str) and SIGNATURE_KEY_RE.match(s)]

    baseline = trusted_dir / MEMORY_SNAPSHOT / KNOWLEDGE_FILE
    knowledge = parse_knowledge(baseline.read_text(encoding="utf-8")) if baseline.is_file() else empty_knowledge()
    notes = read_notes(notes_dir, signatures, warn)
    for group in groups:
        signature = group.get("signature")
        if signature not in signatures:
            continue
        attempt = make_attempt(run_id, str(run.get("url") or ""), remediator_run_url, date, notes.get(signature))
        record_group(knowledge, group, attempt)
    evicted = evict_oldest(knowledge, set(signatures))

    text = dump_knowledge(knowledge)
    parse_knowledge(text)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / KNOWLEDGE_FILE).write_text(text, encoding="utf-8")
    return {"investigated": len(signatures), "evicted": evicted, "signatures": len(knowledge["signatures"])}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("snapshot", "update"):
        command = sub.add_parser(name)
        command.add_argument("--trusted-dir", type=Path, required=True)
        command.add_argument("--notes-dir", type=Path, required=True)
        if name == "snapshot":
            command.add_argument("--digest-dir", type=Path, required=True)
            command.add_argument("--memory-dir", type=Path, required=True)
        else:
            command.add_argument("--out-dir", type=Path, required=True)
            command.add_argument("--remediator-run-url", default="")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "snapshot":
            snapshot(args.digest_dir, args.memory_dir, args.trusted_dir, args.notes_dir)
            print(f"Snapshot of the digest and knowledge base written to {args.trusted_dir}")
        else:
            today = datetime.now(timezone.utc).date().isoformat()
            result = update(
                args.trusted_dir, args.notes_dir, args.out_dir, args.remediator_run_url, today, warn=lambda m: print(f"::warning::{clean_text(m, 300)}")
            )
            if result["signatures"] is None:
                print("No failure groups were investigated; the knowledge base is unchanged")
            else:
                print(
                    f"Recorded {result['investigated']} investigated signatures; the knowledge base holds {result['signatures']} (evicted {len(result['evicted'])})"
                )
    except (KnowledgeError, OSError, ValueError) as error:
        print(f"::error::knowledge base {args.command} failed: {clean_text(str(error), 300)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
