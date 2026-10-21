"""Discover notebooks in the repository, shuffle them, and split into batches.

Outputs newline-separated notebook paths for each batch as GitHub Actions
multiline outputs (batch_0 … batch_N-1). With --plan_file, the split is also
written as JSON for the nightly dashboard (nightly_report/schema/plan.schema.json).

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
    notebooks = sorted(
        str(p) for p in notebooks_dir.rglob("*.ipynb") if not p.name.startswith("test_") and not {".venv", ".ipynb_checkpoints"}.intersection(p.parts)
    )
    return notebooks


def select_precommit_notebooks(notebooks_dir: Path, changed_files: list[str], fallback_file: Path) -> list[str]:
    notebooks = collect_notebooks(notebooks_dir)
    available = {Path(path).resolve(): path for path in notebooks}
    root = notebooks_dir.resolve()
    changed = [Path(path).resolve() for path in changed_files]
    if root.parent / "requirements.txt" in changed:
        if not notebooks:
            raise ValueError("No notebooks found for requirements.txt validation")
        return notebooks

    selected = set()
    for path in changed:
        if not path.is_relative_to(root) or {".venv", ".ipynb_checkpoints"}.intersection(path.parts):
            continue
        if path in available:
            selected.add(available[path])
        elif path.suffix == ".py":
            directory = path.parent
            while directory.is_relative_to(root):
                siblings = [value for notebook, value in available.items() if notebook.parent == directory]
                if siblings:
                    selected.update(siblings)
                    break
                if directory == root:
                    break
                directory = directory.parent
    if selected:
        return sorted(selected)

    fallback = [Path(line.strip()).resolve() for line in fallback_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not fallback or any(path not in available for path in fallback):
        raise ValueError(f"Fallback list must contain existing notebooks: {fallback_file}")
    return sorted({available[path] for path in fallback})


def write_precommit_output(batches: list[list[str]]) -> None:
    matrix = [{"name": f"batch_{index}", "test_list": "\n".join(batch)} for index, batch in enumerate(batches)]
    output = f"matrix={json.dumps(matrix)}\nhas_tests={'true' if matrix else 'false'}\n"
    if output_file := os.environ.get("GITHUB_OUTPUT"):
        with open(output_file, "a", encoding="utf-8") as stream:
            stream.write(output)
    else:
        print(output, end="")
    for batch in matrix:
        print(f"{batch['name']}: {len(batch['test_list'].splitlines())} notebooks")


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
    parser.add_argument("--precommit", action="store_true", help="Select CHANGED_FILES JSON paths and emit up to --num_batches nonempty matrix entries")
    parser.add_argument(
        "--fallback_file", type=Path, default=Path(".ci/precommit_list.txt"), help="Smoke-test list used when no affected notebooks are selected"
    )
    args = parser.parse_args()
    if args.num_batches < 1:
        parser.error("--num_batches must be positive")

    if args.precommit:
        changed_files = json.loads(os.environ["CHANGED_FILES"])
        if not isinstance(changed_files, list) or any(not isinstance(path, str) for path in changed_files):
            raise ValueError("CHANGED_FILES must be a JSON array of paths")
        notebooks = select_precommit_notebooks(args.notebooks_dir, changed_files, args.fallback_file)
    else:
        notebooks = collect_notebooks(args.notebooks_dir)
    print(f"Found {len(notebooks)} notebooks")

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(notebooks)

    batches = split_into_batches(notebooks, min(args.num_batches, len(notebooks)) if args.precommit else args.num_batches)
    if args.precommit:
        write_precommit_output(batches)
    else:
        write_github_output(batches)
    if args.plan_file is not None:
        # Optional dashboard data: never fail the split (all nightly test jobs depend on its outputs)
        try:
            write_plan_file(args.plan_file, batches, args.notebooks_dir, args.seed)
        except Exception as e:
            print(f"WARNING: failed to write batch plan to {args.plan_file}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
