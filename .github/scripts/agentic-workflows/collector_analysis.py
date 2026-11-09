"""Error grouping, remediation tracking, and cross-night history analysis."""

from __future__ import annotations

import hashlib
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from collector_common import (
    INFRA_NOT_RUN_KINDS,
    PR_CANDIDATE_CLASSES,
    REJECTED_COOLDOWN,
    SIGNATURE_RE,
    TRACKED_LIST_LIMIT,
    TRACKING_LABEL,
    CollectorError,
    GhError,
    Group,
    Member,
    Night,
    NoWork,
    TrackedItem,
    TrackingDecision,
    Key,
)
from collector_github import check_schema, download_report, list_json, list_nightly_runs


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infra_signature(reason: str) -> str:
    return hashlib.sha1(f"infra|{reason}".encode("utf-8")).hexdigest()[:12]


def build_groups(report: dict) -> List[Group]:
    """Failure groups keyed by signature, plus infra groups for notebooks that never ran because a job broke."""
    canonical = {g.get("signature"): g for g in report.get("error_groups") or []}
    groups: Dict[str, Group] = {}
    for failure in report.get("failures") or []:
        signature, notebook, config_id = failure.get("signature"), failure.get("notebook"), failure.get("config_id")
        if not (signature and notebook and config_id):
            continue
        group = groups.get(signature)
        if group is None:
            header = canonical.get(signature) or failure
            group = groups[signature] = Group(
                signature=signature,
                kind="error",
                category=header.get("category") or failure.get("category") or "unknown",
                error_class=header.get("error_class") or failure.get("error_class") or "other",
                exception_type=header.get("exception_type") or failure.get("exception_type") or "",
                exception_message=header.get("exception_message") or failure.get("exception_message") or "",
            )
        group.members.append(Member(notebook, config_id, failure.get("status") or "failed", to_int(failure.get("job_id")), failure=failure))

    for result in report.get("results") or []:
        reason = result.get("not_run_reason") or ""
        kind, _, detail = reason.partition(":")
        if result.get("status") != "not_run" or kind not in INFRA_NOT_RUN_KINDS or not (result.get("notebook") and result.get("config_id")):
            continue
        signature = infra_signature(reason)
        group = groups.get(signature)
        if group is None:
            group = groups[signature] = Group(
                signature=signature, kind="infra", category="infra", error_class=kind, exception_type="", exception_message=detail
            )
        group.members.append(Member(result["notebook"], result["config_id"], "not_run", to_int(result.get("job_id")), not_run_reason=reason))
    return list(groups.values())


def rank_key(group: Group) -> tuple:
    return (-len(group.notebooks), -len(group.members), group.signature)


def extract_signature(*texts: Optional[str]) -> Optional[str]:
    for text in texts:
        match = SIGNATURE_RE.search(text or "")
        if match:
            return match.group(1).lower()
    return None


def item_state(kind: str, raw: dict) -> str:
    state = str(raw.get("state", "")).upper()
    if state == "OPEN":
        return "open"
    if kind == "pr":
        return "merged" if state == "MERGED" or raw.get("mergedAt") else "closed"
    return {"NOT_PLANNED": "not_planned", "DUPLICATE": "duplicate"}.get(str(raw.get("stateReason") or "").upper(), "completed")


def fetch_tracked(repo: str, warn: Callable[[str], None]) -> Dict[str, List[TrackedItem]]:
    """Remediation PRs and issues of `repo` (label `nightly-remediator`) by the error signature printed in their title or body."""
    limit = str(TRACKED_LIST_LIMIT)
    common = ["--repo", repo, "--label", TRACKING_LABEL, "--state", "all", "--limit", limit]
    prs = list_json(["pr", "list", *common, "--json", "number,title,body,state,url,createdAt,closedAt,mergedAt"])
    try:
        issues = list_json(["issue", "list", *common, "--json", "number,title,body,state,stateReason,url,createdAt,closedAt"])
    except GhError as error:
        if "disabled issues" not in error.stderr:
            raise
        warn(f"issues are disabled in {repo}; only pull requests are checked for already tracked errors")
        issues = []
    tracked: Dict[str, List[TrackedItem]] = defaultdict(list)
    for kind, items in (("pr", prs), ("issue", issues)):
        for raw in items:
            signature = extract_signature(raw.get("title"), raw.get("body"))
            if signature:
                tracked[signature].append(
                    TrackedItem(
                        kind,
                        raw["number"],
                        raw.get("url", ""),
                        raw.get("title", ""),
                        item_state(kind, raw),
                        signature,
                        raw.get("createdAt", ""),
                        raw.get("closedAt") or raw.get("mergedAt"),
                    )
                )
    for items in tracked.values():
        items.sort(key=lambda item: item.created_at, reverse=True)
    return dict(tracked)


def decide_tracking(items: List[TrackedItem], now: datetime) -> TrackingDecision:
    """Open items and items rejected within the cooldown block the group; merged or completed ones are passed on as previous attempts."""
    blocking: Optional[TrackedItem] = None
    reason = ""
    for item in items:
        if item.state == "open":
            blocking, reason = item, "open"
            break
        closed_at = parse_ts(item.closed_at)
        if item.state in ("closed", "not_planned", "duplicate") and closed_at and now - closed_at < REJECTED_COOLDOWN and blocking is None:
            blocking, reason = item, "recently rejected"
    return TrackingDecision(skip=blocking is not None, reason=reason, blocking=blocking, previous=[i for i in items if i is not blocking][:5])


def is_pr_candidate(group: Group) -> bool:
    return group.kind == "error" and group.error_class in PR_CANDIDATE_CLASSES


def select_groups(groups: List[Group], tracked: Dict[str, List[TrackedItem]], max_groups: int, max_prs: int, now: datetime):
    """Picks the groups for the agent; tracked groups do not use a slot.

    Groups that a PR can plausibly fix come first (up to `max_prs`), so that wide but issue-only groups (infra, timeouts, crashes) cannot
    take every slot. The remaining slots go to the widest other groups, then to further PR candidates. Within each kind the order is by
    affected notebooks, then occurrences. Everything not picked is deferred to later nights.
    """
    skipped: List[Tuple[Group, TrackingDecision]] = []
    fixable: List[Tuple[Group, TrackingDecision]] = []
    other: List[Tuple[Group, TrackingDecision]] = []
    for group in sorted(groups, key=rank_key):
        decision = decide_tracking(tracked.get(group.signature, []), now)
        (skipped if decision.skip else fixable if is_pr_candidate(group) else other).append((group, decision))
    quota = min(len(fixable), max_prs, max_groups)
    chosen = fixable[:quota] + other[: max_groups - quota]
    chosen += fixable[quota:][: max_groups - len(chosen)]
    chosen_signatures = {group.signature for group, _ in chosen}
    selected = [item for item in fixable + other if item[0].signature in chosen_signatures]
    deferred = [group for group, _ in fixable + other if group.signature not in chosen_signatures]
    return selected, skipped, deferred


def night_from_report(report: dict) -> Night:
    return Night(
        info=report.get("run") or {},
        signatures={g.signature: {m.key for m in g.members} for g in build_groups(report)},
        statuses={(r["notebook"], r["config_id"]): r.get("status") or "" for r in report.get("results") or [] if r.get("notebook") and r.get("config_id")},
    )


def fetch_history(source_repo: str, current: dict, count: int, workdir: Path, warn: Callable[[str], None]) -> List[Night]:
    """Reports of the previous scheduled nightlies, newest first; best effort."""
    if count <= 0:
        return []
    try:
        runs = list_nightly_runs(source_repo, per_page=30)
    except CollectorError as error:
        warn(f"history unavailable: {error}")
        return []
    older = sorted(
        (r for r in runs if r["id"] != current["id"] and r["created_at"] < current["created_at"] and r.get("conclusion") in ("success", "failure")),
        key=lambda r: r["created_at"],
        reverse=True,
    )
    nights: List[Night] = []
    for run in older[: count * 2]:
        if len(nights) >= count:
            break
        try:
            report, _ = download_report(source_repo, run["id"], workdir, with_environments=False)
            check_schema(report)
        except (CollectorError, NoWork, ValueError, zipfile.BadZipFile) as error:
            warn(f"history: skipped run {run['id']}: {error}")
            continue
        nights.append(night_from_report(report))
    return nights


def classify_previous(key: Key, signature: str, night: Night) -> str:
    if key in night.signatures.get(signature, set()):
        return "same_error"
    status = night.statuses.get(key)
    if status == "passed":
        return "passed"
    if status in ("failed", "timeout", "error"):
        return "other_failure"
    return "not_run_or_absent"


def history_rows(group: Group, current: Night, previous: List[Night]) -> List[dict]:
    keys = {m.key for m in group.members}
    rows = []
    for index, night in enumerate([current, *previous]):
        counts = Counter(classify_previous(key, group.signature, night) for key in keys)
        info = night.info
        rows.append(
            {
                "current": index == 0,
                "run_number": info.get("number"),
                "run_url": info.get("url"),
                "date": info.get("nightly_date") or str(info.get("created_at", ""))[:10],
                "conclusion": info.get("conclusion"),
                "commit": (info.get("commit") or {}).get("short_sha"),
                "docker_tag": (info.get("inputs") or {}).get("docker_tag"),
                "same_error": counts["same_error"],
                "passed": counts["passed"],
                "other_failure": counts["other_failure"],
                "not_run_or_absent": counts["not_run_or_absent"],
            }
        )
    return rows


def trend_label(signature: str, previous: List[Night]) -> str:
    if not previous:
        return "unknown (no history)"
    flags = [signature in night.signatures for night in previous]
    seen, total = 1 + sum(flags), 1 + len(previous)
    if not flags[0]:
        return "new tonight" if seen == 1 else f"intermittent ({seen}/{total} nights)"
    return f"recurring (all {total} nights)" if seen == total else f"recurring ({seen}/{total} nights)"
