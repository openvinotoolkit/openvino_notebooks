"""Build the nightly dashboard report from the artifacts of a treon_nightly workflow run.

Inputs:
  * plan.json written by split_notebooks.py (--plan_file): batch membership of notebooks;
  * fragment.json (+ logs/, envs/) of every test job written by validate_notebooks.py (--dashboard_dir);
  * GitHub REST API: run metadata, jobs (conclusions, failed steps, links) and artifacts (log links);
  * the repository checkout: notebook catalog (title, tags) and git metadata.

Outputs (--output_dir): nightly-report.json, environments.json, summary.md and a copy of the report schema.

Data problems (missing fragments, API errors, unparsable logs) never stop the build; they are recorded in
generator.warnings. The script exits with a non-zero code only if the produced report violates the schema.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nightly_report.constants import ENVIRONMENTS_FILE_NAME, REPORT_FILE_NAME, SUMMARY_FILE_NAME  # noqa: E402
from nightly_report.report_builder import ReportBuilder  # noqa: E402
from nightly_report.report_inputs import REPORT_SCHEMA_PATH, get_validator, schema_errors  # noqa: E402
from nightly_report.summary_md import render_markdown  # noqa: E402


def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fragments_dir", type=Path, required=True, help="Directory with downloaded nightly-fragment-* artifacts.")
    parser.add_argument("--plan", type=Path, help="plan.json written by split_notebooks.py --plan_file.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--notebooks_dir", default="notebooks", help="Notebooks directory relative to the repository root.")
    parser.add_argument("--skip_config", default=".ci/skipped_notebooks.yml", help="Skip config relative to the repository root.")
    parser.add_argument("--ignore_list", nargs="*", default=[], help="Ignore lists passed to validate_notebooks.py (repo-root relative).")
    parser.add_argument("--ov_branch")
    parser.add_argument("--docker_tag")
    parser.add_argument("--timeout", type=float, help="Per-notebook timeout in seconds.")
    parser.add_argument("--latest_ref", default="latest", help="Branch used for 'latest' notebook links.")
    parser.add_argument(
        "--workflow_file",
        help="Workflow file (repo-root relative) with the test job matrices; used to report jobs skipped or cancelled before matrix expansion.",
    )
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", "openvinotoolkit/openvino_notebooks"))
    parser.add_argument("--run_id", type=int, default=int(os.environ.get("GITHUB_RUN_ID", "0") or 0))
    parser.add_argument("--run_attempt", type=int, default=int(os.environ.get("GITHUB_RUN_ATTEMPT", "1") or 1))
    parser.add_argument("--server_url", default=os.environ.get("GITHUB_SERVER_URL", "https://github.com"))
    parser.add_argument("--api_url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    parser.add_argument("--no_api", action="store_true", help="Do not call the GitHub API.")
    parser.add_argument("--run_json", type=Path, help="Use this file instead of GET /actions/runs/{id}.")
    parser.add_argument("--jobs_json", type=Path, help="Use this file ({'jobs': [...]}) instead of the jobs API.")
    parser.add_argument("--artifacts_json", type=Path, help="Use this file ({'artifacts': [...]}) instead of the artifacts API.")
    parser.add_argument("--job_logs_dir", type=Path, help="Use <job id>.log files of this directory instead of the job logs API.")
    parser.add_argument("--summary_file", type=Path, help="Append the markdown summary to this file (e.g. $GITHUB_STEP_SUMMARY).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_arguments(argv)
    builder = ReportBuilder(args)
    report, environments = builder.build()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / REPORT_FILE_NAME).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / ENVIRONMENTS_FILE_NAME).write_text(json.dumps(environments, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copyfile(REPORT_SCHEMA_PATH, args.output_dir / REPORT_SCHEMA_PATH.name)

    markdown = render_markdown(report)
    (args.output_dir / SUMMARY_FILE_NAME).write_text(markdown, encoding="utf-8")
    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as summary_file:
            summary_file.write(markdown + "\n")

    summary = report["summary"]
    print(
        f"Report written to '{args.output_dir / REPORT_FILE_NAME}': {len(report['results'])} results, {len(report['failures'])} failures, "
        f"pass rate {summary['pass_rate']}, coverage {summary['coverage']}, {len(builder.warnings.items)} warning(s)."
    )

    errors = schema_errors(get_validator(REPORT_SCHEMA_PATH), report)
    if errors:
        print("ERROR: the report does not match the schema:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
