#!/usr/bin/env python3
"""Pre-agent step of the nightly issue remediator (no AI, stdlib + the `gh` CLI).

Downloads the `nightly-dashboard-data` report of a `treon_nightly` run, groups the failures by error signature, drops groups that already
have an open (or recently rejected) remediation PR/issue, ranks the rest, downloads the raw logs of the most representative failures
and writes a digest that the agent reads instead of calling GitHub itself.

Needs `GH_TOKEN` with `actions: read` on the source repository and `issues: read` + `pull-requests: read` on `--repo`.
The report format is described in `.ci/nightly_report/README.md`.

Digest layout (`--out`):
  status.json    `has_work` flag, reason and counts (the workflow skips the agent when `has_work` is false)
  summary.md     entry point: run, ranked groups, already tracked and deferred groups
  run.json       run metadata from the report
  tracked.json   remediation PRs/issues found per error signature
  groups/NN-<signature>/group.md    evidence of one error group: where it starts, affected notebooks, history, versions, excerpts
  groups/NN-<signature>/group.json  the same in machine-readable form
  groups/NN-<signature>/logs/       raw treon logs of representative failures (or job log extracts for infra groups)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from collector_common import ALLOWED_SOURCE_REPOS, REPO_RE, GroupDigest, NoWork
from collector_github import check_schema, download_report, resolve_run
from collector_analysis import build_groups, fetch_history, fetch_tracked, night_from_report, select_groups
from collector_evidence import make_context
from collector_render import build_digest, group_to_json, render_group, render_summary


def validate_args(args: argparse.Namespace) -> None:
    for name in ("repo", "source_repo"):
        if not REPO_RE.match(getattr(args, name)):
            raise SystemExit(f"--{name.replace('_', '-')} must look like owner/name")
    allowed = {args.repo.lower(), *(r.lower() for r in ALLOWED_SOURCE_REPOS)}
    if args.source_repo.lower() not in allowed:
        raise SystemExit(f"--source-repo {args.source_repo} is not allowed (allowed: {', '.join(sorted(allowed))})")
    if args.run_id and not re.fullmatch(r"\d{1,20}", args.run_id):
        raise SystemExit("--run-id must be a number")
    for name, low, high in (("max_groups", 1, 10), ("max_prs", 1, 10), ("history_nights", 0, 7), ("logs_per_group", 1, 6)):
        if not low <= getattr(args, name) <= high:
            raise SystemExit(f"--{name.replace('_', '-')} must be between {low} and {high}")


def prepare_out_dir(path: str) -> Path:
    out = Path(path).resolve()
    if out in (Path("/"), Path.home().resolve(), Path.cwd().resolve()) or len(out.parts) < 3:
        raise SystemExit(f"refusing to use {out} as the digest directory")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finish(out: Path, status: dict, summary: Optional[str] = None) -> None:
    write_json(out / "status.json", status)
    if summary is None:
        summary = f"# Nightly failures digest\n\n{status['reason']}\n"
    (out / "summary.md").write_text(summary, encoding="utf-8")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"has_work={'true' if status['has_work'] else 'false'}\ngroups_selected={status.get('groups_selected', 0)}\n")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")


def collect(args: argparse.Namespace, out: Path, warnings: List[str], now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)

    def warn(message: str) -> None:
        warnings.append(message)
        print(f"::warning::{message}")

    with tempfile.TemporaryDirectory(prefix="nightly-remediator-") as tmp:
        workdir = Path(tmp)
        run = resolve_run(args.source_repo, args.run_id)
        print(f"Using {args.source_repo} run {run['id']} (conclusion: {run.get('conclusion')}, created: {run.get('created_at')})")
        report, environments = download_report(args.source_repo, run["id"], workdir, with_environments=True)
        check_schema(report)
        ctx = make_context(args.repo, args.source_repo, report, environments)
        write_json(out / "run.json", {**ctx.run, "source_repository": args.source_repo, "report_schema_version": report.get("schema_version")})

        groups = build_groups(report)
        status: dict = {
            "has_work": False,
            "reason": "",
            "run_id": run["id"],
            "run_number": ctx.run.get("number"),
            "run_url": ctx.run.get("url"),
            "failures_total": len(report.get("failures") or []),
            "error_groups_total": sum(1 for g in groups if g.kind == "error"),
            "infra_groups_total": sum(1 for g in groups if g.kind == "infra"),
            "infra_notebooks_not_run": sum(len(g.members) for g in groups if g.kind == "infra"),
            "groups_selected": 0,
            "groups_tracked": 0,
            "groups_deferred": 0,
            "warnings": warnings,
        }
        if not groups:
            status["reason"] = "the nightly report contains no failures and no infra problems"
            finish(out, status, render_summary(ctx, status, [], [], [], args))
            return status

        tracked = fetch_tracked(args.repo, warn, now)
        write_json(out / "tracked.json", {signature: [item.to_json() for item in items] for signature, items in tracked.items()})
        selected, skipped, deferred = select_groups(groups, tracked, args.max_groups, args.max_prs, now)
        status.update(groups_selected=len(selected), groups_tracked=len(skipped), groups_deferred=len(deferred))
        if not selected:
            status["reason"] = f"all {len(skipped)} failure groups are already tracked by open or recently rejected remediation PRs/issues"
            finish(out, status, render_summary(ctx, status, [], skipped, deferred, args))
            return status

        current = night_from_report(report)
        previous = fetch_history(args.source_repo, run, args.history_nights, workdir, warn)
        digests: List[GroupDigest] = []
        used_bytes = [0]
        for index, (group, decision) in enumerate(selected, start=1):
            group_dir = out / "groups" / f"{index:02d}-{group.signature}"
            group_dir.mkdir(parents=True)
            digest = build_digest(ctx, index, group, decision, current, previous, args, group_dir, workdir, used_bytes, warn)
            (group_dir / "group.md").write_text(render_group(ctx, digest, previous), encoding="utf-8")
            write_json(group_dir / "group.json", group_to_json(ctx, digest))
            digests.append(digest)

        status["has_work"] = True
        status["reason"] = f"{len(selected)} failure groups selected for remediation, {len(skipped)} already tracked, {len(deferred)} deferred"
        finish(out, status, render_summary(ctx, status, digests, skipped, deferred, args))
        return status


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True, help="repository that receives the PRs and issues (owner/name)")
    parser.add_argument("--source-repo", help="repository whose nightly is analysed (default: --repo)")
    parser.add_argument("--run-id", default="", help="nightly run id (default: the latest completed scheduled run)")
    parser.add_argument("--max-groups", type=int, default=5, help="error groups handed to the agent")
    parser.add_argument("--max-prs", type=int, default=3, help="groups that a PR can plausibly fix and that get a slot before issue-only groups")
    parser.add_argument("--history-nights", type=int, default=3, help="previous nightlies used for history")
    parser.add_argument("--logs-per-group", type=int, default=3, help="raw logs downloaded per error group")
    parser.add_argument("--out", required=True, help="digest directory (recreated)")
    args = parser.parse_args(argv)
    args.source_repo = args.source_repo or args.repo
    args.run_id = args.run_id.strip()
    validate_args(args)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = prepare_out_dir(args.out)
    warnings: List[str] = []
    try:
        status = collect(args, out, warnings)
    except NoWork as reason:
        print(f"::warning::{reason}")
        finish(out, {"has_work": False, "reason": str(reason), "groups_selected": 0, "warnings": warnings})
        return 0
    except Exception as error:  # the digest must say why there is none; the job still fails so that the problem is visible
        traceback.print_exc()
        print(f"::error::nightly failure collection failed: {error}")
        finish(out, {"has_work": False, "reason": f"collector failed: {error}", "groups_selected": 0, "warnings": warnings})
        return 1
    print(f"{status['reason']} (has_work={status['has_work']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
