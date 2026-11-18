"""Knowledge base of earlier nightly remediation attempts, persisted per error signature on a dedicated repository branch.

The workflow's `repo-memory` tool keeps `knowledge.json` on `KNOWLEDGE_BRANCH`. Only trusted scripts write it, and every reader validates it strictly:
its text was derived from untrusted CI data and agent notes, so it is evidence for an investigation and never an instruction.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

KNOWLEDGE_BRANCH = "memory/nightly-remediator"
KNOWLEDGE_FILE = "knowledge.json"
SCHEMA_VERSION = 1
MAX_SIGNATURES = 100
MAX_ATTEMPTS = 3
MAX_NOTEBOOKS = 25
MAX_FILE_BYTES = 900_000
TEXT_LIMIT = 400
URL_LIMIT = 200
SIGNATURE_KEY_RE = re.compile(r"^[0-9a-f]{12}$")
DATE_RE = re.compile(r"^\d{4}-\d\d-\d\d$")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
URL_PREFIX = "https://github.com/"
OUTCOMES = ("pr", "issue", "none")
CONFIDENCES = ("", "low", "medium", "high")
NOTE_FIELDS = ("root_cause", "action", "dead_ends", "next_step")
RECORD_KEYS = {"kind", "category", "error_class", "exception_type", "headline", "first_seen", "last_seen", "investigations", "notebooks", "attempts"}
ATTEMPT_KEYS = {"nightly_run_id", "nightly_run_url", "remediator_run_url", "date", "outcome", "confidence", *NOTE_FIELDS}


class KnowledgeError(ValueError):
    """The stored knowledge base does not match the schema."""


def clean_text(value: Any, limit: int = TEXT_LIMIT) -> str:
    """One bounded line without control characters; anything but a string is empty."""
    if not isinstance(value, str):
        return ""
    text = " ".join(CONTROL_RE.sub(" ", value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def clean_url(value: Any) -> str:
    return (
        value
        if isinstance(value, str) and value.startswith(URL_PREFIX) and len(value) <= URL_LIMIT and not CONTROL_RE.search(value) and " " not in value
        else ""
    )


def empty_knowledge() -> dict:
    return {"schema_version": SCHEMA_VERSION, "signatures": {}}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KnowledgeError(message)


def _check_text(value: Any, where: str, limit: int = TEXT_LIMIT) -> None:
    _require(isinstance(value, str) and len(value) <= limit and not CONTROL_RE.search(value), f"{where} must be a short single-line string")


def validate_record(signature: str, record: Any) -> None:
    where = f"signature {signature}"
    _require(isinstance(record, dict) and set(record) == RECORD_KEYS, f"{where} has unexpected fields")
    for name in ("kind", "category", "error_class", "exception_type", "headline"):
        _check_text(record[name], f"{where} {name}")
    _require(record["kind"] in ("error", "infra"), f"{where} has an invalid kind")
    for name in ("first_seen", "last_seen"):
        _require(isinstance(record[name], str) and DATE_RE.match(record[name]) is not None, f"{where} {name} must be a YYYY-MM-DD date")
    _require(type(record["investigations"]) is int and record["investigations"] >= 1, f"{where} has an invalid investigations count")
    notebooks = record["notebooks"]
    _require(isinstance(notebooks, list) and len(notebooks) <= MAX_NOTEBOOKS, f"{where} has too many notebooks")
    for notebook in notebooks:
        _check_text(notebook, f"{where} notebook")
    attempts = record["attempts"]
    _require(isinstance(attempts, list) and 1 <= len(attempts) <= MAX_ATTEMPTS, f"{where} has an invalid number of attempts")
    for attempt in attempts:
        _require(isinstance(attempt, dict) and set(attempt) == ATTEMPT_KEYS, f"{where} has an attempt with unexpected fields")
        _require(type(attempt["nightly_run_id"]) is int and attempt["nightly_run_id"] >= 0, f"{where} has an invalid nightly run id")
        for name in ("nightly_run_url", "remediator_run_url"):
            _require(attempt[name] == clean_url(attempt[name]), f"{where} has an invalid {name}")
        _require(isinstance(attempt["date"], str) and DATE_RE.match(attempt["date"]) is not None, f"{where} has an invalid attempt date")
        _require(attempt["outcome"] in OUTCOMES, f"{where} has an invalid outcome")
        _require(attempt["confidence"] in CONFIDENCES, f"{where} has an invalid confidence")
        for name in NOTE_FIELDS:
            _check_text(attempt[name], f"{where} {name}")


def parse_knowledge(text: str) -> dict:
    """Strictly parses a stored knowledge base; any deviation from the schema is an error."""
    try:
        data = json.loads(text)
    except ValueError as error:
        raise KnowledgeError(f"{KNOWLEDGE_FILE} is not valid JSON: {error}") from error
    _require(isinstance(data, dict) and set(data) == {"schema_version", "signatures"}, f"{KNOWLEDGE_FILE} has unexpected top-level fields")
    _require(data["schema_version"] == SCHEMA_VERSION and type(data["schema_version"]) is int, f"unsupported {KNOWLEDGE_FILE} schema version")
    signatures = data["signatures"]
    _require(isinstance(signatures, dict) and len(signatures) <= MAX_SIGNATURES, f"{KNOWLEDGE_FILE} holds more than {MAX_SIGNATURES} signatures")
    for signature, record in signatures.items():
        _require(SIGNATURE_KEY_RE.match(signature) is not None, f"invalid signature key {signature!r}")
        validate_record(signature, record)
    return data


def dump_knowledge(knowledge: dict) -> str:
    return json.dumps(knowledge, indent=1, sort_keys=True) + "\n"


def make_attempt(run_id: int, nightly_run_url: str, remediator_run_url: str, date: str, note: Optional[dict]) -> dict:
    """One investigation; the agent's note supplies the outcome it submitted, its confidence and the free-text fields."""
    note = note or {}
    confidence = note.get("confidence")
    outcome = note.get("outcome")
    attempt: Dict[str, Any] = {
        "nightly_run_id": run_id,
        "nightly_run_url": clean_url(nightly_run_url),
        "remediator_run_url": clean_url(remediator_run_url),
        "date": date,
        "outcome": outcome if outcome in OUTCOMES else "none",
        "confidence": confidence if confidence in CONFIDENCES else "",
    }
    for name in NOTE_FIELDS:
        attempt[name] = clean_text(note.get(name))
    return attempt


def record_group(knowledge: dict, group: dict, attempt: dict) -> None:
    """Folds one investigation of `group` (a digest `group.json`) into its signature record."""
    signature = group["signature"]
    date = attempt["date"]
    existing = knowledge["signatures"].get(signature)
    previous = [a for a in (existing["attempts"] if existing else []) if a["nightly_run_id"] != attempt["nightly_run_id"]]
    repeated = existing is not None and len(previous) != len(existing["attempts"])
    kind = group.get("kind") if group.get("kind") in ("error", "infra") else "error"
    category = clean_text(group.get("category"), 80)
    error_class = clean_text(group.get("error_class"), 80)
    exception_type = clean_text(group.get("exception_type"), 120)
    message = clean_text(group.get("exception_message"))
    if kind == "infra":
        headline = f"{error_class}: {message}" if message else error_class
    else:
        headline = f"{exception_type}: {message}" if exception_type else message or category
    knowledge["signatures"][signature] = {
        "kind": kind,
        "category": category,
        "error_class": error_class,
        "exception_type": exception_type,
        "headline": clean_text(headline, 300),
        "first_seen": min(existing["first_seen"], date) if existing else date,
        "last_seen": max(existing["last_seen"], date) if existing else date,
        "investigations": (existing["investigations"] if existing else 0) + (0 if repeated else 1),
        "notebooks": [clean_text(n, 200) for n in sorted({str(n) for n in group.get("notebooks") or []})][:MAX_NOTEBOOKS],
        "attempts": ([attempt] + previous)[:MAX_ATTEMPTS],
    }


def evict_oldest(knowledge: dict, protected: set) -> List[str]:
    """Drops the least recently seen signatures until the knowledge base fits its signature and size caps."""
    evicted: List[str] = []
    signatures = knowledge["signatures"]
    while len(signatures) > MAX_SIGNATURES or len(dump_knowledge(knowledge).encode("utf-8")) > MAX_FILE_BYTES:
        candidates = [s for s in signatures if s not in protected]
        if not candidates:
            raise KnowledgeError("the signatures investigated in this run alone exceed the knowledge base size cap")
        oldest = min(candidates, key=lambda s: (signatures[s]["last_seen"], s))
        del signatures[oldest]
        evicted.append(oldest)
    return evicted


def related_records(knowledge: dict, signature: str, exception_type: str, notebooks: List[str], limit: int = 3) -> List[dict]:
    """Records of other signatures that share the exception type or an affected notebook, most recently seen first."""
    notebook_set = set(notebooks)
    related: List[dict] = []
    for other, record in knowledge["signatures"].items():
        if other == signature:
            continue
        reasons = []
        if exception_type and record["exception_type"] == exception_type:
            reasons.append("same exception type")
        shared = sorted(notebook_set.intersection(record["notebooks"]))
        if shared:
            reasons.append(f"shares {shared[0]}" + (f" (+{len(shared) - 1} more)" if len(shared) > 1 else ""))
        if reasons:
            related.append({"signature": other, "reasons": reasons, "record": record})
    related.sort(key=lambda item: (item["record"]["last_seen"], item["signature"]), reverse=True)
    return related[:limit]
