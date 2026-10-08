"""Discover notebooks in the repository, shuffle them, and split into batches.

Outputs newline-separated notebook paths for each batch as GitHub Actions
multiline outputs (batch_0 … batch_N-1). With --plan_file, the split is also
written as JSON for the nightly dashboard (nightly_report/schema/plan.schema.json).
With --notebooks_list, only the listed notebooks are split instead of all of them.

Usage (in a workflow step):
    python .ci/split_notebooks.py --notebooks_dir notebooks \
                                  --num_batches 4 \
                                  --seed "${{ github.run_id }}"
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional


def collect_notebooks(notebooks_dir: Path) -> list[str]:
    """Return sorted list of notebook paths (relative to repo root), excluding test_ prefixed files."""
    notebooks = sorted(str(p) for p in notebooks_dir.rglob("*.ipynb") if not p.name.startswith("test_"))
    return notebooks


def read_notebooks_list(list_file: Path, notebooks_dir: Path) -> list[str]:
    """Return sorted notebook paths listed in *list_file*: one path relative to the repo root per line, '#' starts a comment.

    Paths are returned in the format of collect_notebooks(). Raises ValueError for entries that are not notebooks inside *notebooks_dir*.
    """
    root = Path(notebooks_dir).resolve()
    notebooks: set[str] = set()
    invalid: list[str] = []
    for raw_line in Path(list_file).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        path = Path(line).resolve()
        if path.suffix != ".ipynb" or path.name.startswith("test_") or not path.is_file() or not path.is_relative_to(root):
            invalid.append(line)
            continue
        notebooks.add(str(Path(notebooks_dir) / path.relative_to(root)))
    if invalid:
        raise ValueError(f"Invalid entries in notebooks list '{list_file}' (expected existing notebooks inside '{notebooks_dir}'): {invalid}")
    return sorted(notebooks)


def split_into_batches(items: list[str], num_batches: int) -> list[list[str]]:
    """Round-robin split *items* into *num_batches* lists."""
    batches: list[list[str]] = [[] for _ in range(num_batches)]
    for idx, item in enumerate(items):
        batches[idx % num_batches].append(item)
    return batches


def write_github_output(batches: list[list[str]]) -> None:
    """Write each batch as a multiline GitHub Actions output variable."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        # When running locally, just print to stdout
        for i, batch in enumerate(batches):
            print(f"--- batch_{i} ({len(batch)} notebooks) ---\n")
            print("\n".join(batch))
        return

    with open(output_file, "a") as fh:
        for i, batch in enumerate(batches):
            print(f"Batch {i}: {len(batch)} notebooks")
            print(f"--- Batch {i} ---" + "\n".join(batch))
            fh.write(f"batch_{i}<<BATCH_EOF\n")
            fh.write("\n".join(batch) + "\n")
            fh.write("BATCH_EOF\n")


def write_plan_file(plan_file: Path, batches: list[list[str]], notebooks_dir: Path, seed: Optional[str]) -> None:
    """Write the batch plan (nightly dashboard data, see nightly_report/schema/plan.schema.json)."""
    from nightly_report.constants import SCHEMA_VERSION

    plan = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "num_batches": len(batches),
        "notebooks_dir": Path(notebooks_dir).as_posix(),
        "notebooks_total": sum(len(batch) for batch in batches),
        "batches": {f"batch_{i}": [Path(notebook).as_posix() for notebook in batch] for i, batch in enumerate(batches)},
    }
    plan_file = Path(plan_file)
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    print(f"Batch plan written to {plan_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split notebooks into batches for CI")
    parser.add_argument(
        "--notebooks_dir",
        type=Path,
        default=Path("notebooks"),
        help="Root directory containing notebooks (default: notebooks)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=4,
        help="Number of batches to split notebooks into (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Random seed for shuffling (e.g. github.run_id for reproducibility within a run)",
    )
    parser.add_argument(
        "--plan_file",
        type=Path,
        default=None,
        help="Optional path to write the batch plan JSON to (nightly dashboard data)",
    )
    parser.add_argument(
        "--notebooks_list",
        type=Path,
        default=None,
        help="Optional file listing the notebooks to split (one repo-root relative path per line) instead of all notebooks",
    )
    args = parser.parse_args()

    if args.notebooks_list is not None:
        try:
            notebooks = read_notebooks_list(args.notebooks_list, args.notebooks_dir)
        except (OSError, ValueError) as e:
            parser.error(str(e))
        # An empty batch would make its test job run all notebooks (no test list).
        if len(notebooks) < args.num_batches:
            parser.error(f"Notebooks list '{args.notebooks_list}' has {len(notebooks)} notebook(s), fewer than --num_batches {args.num_batches}.")
        print(f"Using {len(notebooks)} notebooks from {args.notebooks_list}")
    else:
        notebooks = collect_notebooks(args.notebooks_dir)
        print(f"Found {len(notebooks)} notebooks")

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(notebooks)

    batches = split_into_batches(notebooks, args.num_batches)
    write_github_output(batches)
    if args.plan_file is not None:
        # Optional dashboard data: never fail the split (all nightly test jobs depend on its outputs)
        try:
            write_plan_file(args.plan_file, batches, args.notebooks_dir, args.seed)
        except Exception as e:
            print(f"WARNING: failed to write batch plan to {args.plan_file}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
