"""Error grouping, remediation tracking, and cross-night history analysis."""

from __future__ import annotations

import hashlib
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from collector_common import (
    INFRA_NOT_RUN_KINDS,
    PR_CANDIDATE_CLASSES,
    REJECTED_COOLDOWN,
    SIGNATURE_RE,
    TRACKED_HISTORY_LIMIT,
    TRACKING_LABEL,
    CollectorError,
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


def tracking_pages(repo: str, kind: str, closed: bool) -> Iterator[dict]:
    """Fetch labeled items in cursor order without GitHub Search's result limit."""
    owner, name = repo.split("/", 1)
    connection = "pullRequests" if kind == "pr" else "issues"
    states = "CLOSED,MERGED" if kind == "pr" and closed else "CLOSED" if closed else "OPEN"
    order = "UPDATED_AT,direction:DESC" if closed else "CREATED_AT,direction:ASC"
    extra = "mergedAt" if kind == "pr" else "stateReason"
    query = (
        "query($owner:String!,$name:String!,$label:String!,$cursor:String){"
        "repository(owner:$owner,name:$name){hasIssuesEnabled "
        f"{connection}(first:100,after:$cursor,labels:[$label],states:[{states}],orderBy:{{field:{order}}})"
        f"{{nodes{{number title body state url createdAt updatedAt closedAt {extra}}} pageInfo{{hasNextPage endCursor}}}}"
        "}}"
    )
    cursor: Optional[str] = None
    seen_cursors: set[str] = set()
    while True:
        args = [
            "api",
            "graphql",
            "--raw-field",
            f"query={query}",
            "--raw-field",
            f"owner={owner}",
            "--raw-field",
            f"name={name}",
            "--raw-field",
            f"label={TRACKING_LABEL}",
        ]
        if cursor:
            args.extend(["--raw-field", f"cursor={cursor}"])
        response = list_json(args)
        if not isinstance(response, dict):
            raise CollectorError(f"GitHub returned an invalid tracking response for {repo}")
        data = response.get("data")
        repository = data.get("repository") if isinstance(data, dict) else None
        if response.get("errors") or not isinstance(repository, dict):
            raise CollectorError(f"GitHub returned an invalid tracking response for {repo}: {response.get('errors')}")
        page = repository.get(connection)
        if not isinstance(repository.get("hasIssuesEnabled"), bool) or not isinstance(page, dict):
            raise CollectorError(f"GitHub returned an invalid {kind} tracking page for {repo}")
        nodes, page_info = page.get("nodes"), page.get("pageInfo")
        if not isinstance(nodes, list) or not isinstance(page_info, dict) or not isinstance(page_info.get("hasNextPage"), bool):
            raise CollectorError(f"GitHub returned invalid {kind} tracking pagination for {repo}")
        yield repository
        if not page_info["hasNextPage"]:
            return
        next_cursor = page_info.get("endCursor")
        if not nodes or not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
            raise CollectorError(f"GitHub returned an invalid {kind} tracking cursor for {repo}")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def fetch_tracked(repo: str, warn: Callable[[str], None], now: Optional[datetime] = None) -> Dict[str, List[TrackedItem]]:
    """Fetch every blocker and a bounded history of labeled remediations by signature."""
    cutoff = (now or datetime.now(timezone.utc)) - REJECTED_COOLDOWN
    tracked: Dict[str, List[TrackedItem]] = defaultdict(list)
    issues_enabled = True
    seen: set[Tuple[str, int]] = set()
    for kind, connection in (("pr", "pullRequests"), ("issue", "issues")):
        if kind == "issue" and not issues_enabled:
            warn(f"issues are disabled in {repo}; only pull requests are checked for already tracked errors")
            continue
        for closed in (False, True):
            allowed_states = ("CLOSED", "MERGED") if closed and kind == "pr" else ("CLOSED",) if closed else ("OPEN",)
            history_count = 0
            for repository in tracking_pages(repo, kind, closed):
                issues_enabled = repository["hasIssuesEnabled"]
                nodes = repository[connection]["nodes"]
                last_updated: Optional[datetime] = None
                for raw in nodes:
                    if not isinstance(raw, dict) or type(raw.get("number")) is not int or raw["number"] < 1 or raw.get("state") not in allowed_states:
                        raise CollectorError(f"GitHub returned an invalid {kind} tracking item for {repo}")
                    state = item_state(kind, raw)
                    recent_rejection = False
                    if closed:
                        last_updated = parse_ts(raw.get("updatedAt"))
                        if last_updated is None:
                            raise CollectorError(f"closed {kind} #{raw.get('number')} has no valid updatedAt")
                        if state in ("closed", "not_planned", "duplicate"):
                            closed_at = parse_ts(raw.get("closedAt"))
                            if closed_at is None:
                                raise CollectorError(f"rejected {kind} #{raw.get('number')} has no valid closedAt")
                            recent_rejection = closed_at > cutoff
                        if not recent_rejection:
                            history_count += 1
                    if closed and not recent_rejection and history_count > TRACKED_HISTORY_LIMIT:
                        continue
                    key = (kind, raw["number"])
                    if key in seen:
                        continue
                    seen.add(key)
                    signature = extract_signature(raw.get("title"), raw.get("body"))
                    if signature:
                        tracked[signature].append(
                            TrackedItem(
                                kind,
                                raw["number"],
                                raw.get("url", ""),
                                raw.get("title", ""),
                                state,
                                signature,
                                raw.get("createdAt", ""),
                                raw.get("closedAt") or raw.get("mergedAt"),
                            )
                        )
                # A recent rejection must have an updatedAt after the cutoff.
                if closed and history_count >= TRACKED_HISTORY_LIMIT and (last_updated is None or last_updated <= cutoff):
                    break
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
