"""Discover notebooks in the repository, shuffle them, and split into batches.

Outputs newline-separated notebook paths for each batch as GitHub Actions
multiline outputs (batch_0 … batch_N-1).

Usage (in a workflow step):
    python .ci/split_notebooks.py --notebooks_dir notebooks \
                                  --num_batches 4 \
                                  --seed "${{ github.run_id }}"
"""

import argparse
import os
import random
from pathlib import Path


def collect_notebooks(notebooks_dir: Path) -> list[str]:
    """Return sorted list of notebook paths (relative to repo root), excluding test_ prefixed files."""
    notebooks = sorted(
        str(p)
        for p in notebooks_dir.rglob("*.ipynb")
        if not p.name.startswith("test_")
    )
    return notebooks


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
            print(f"--- batch_{i} ({len(batch)} notebooks) ---")
            print("\n".join(batch))
        return

    with open(output_file, "a") as fh:
        for i, batch in enumerate(batches):
            print(f"Batch {i}: {len(batch)} notebooks")
            fh.write(f"batch_{i}<<BATCH_EOF\n")
            fh.write("\n".join(batch) + "\n")
            fh.write("BATCH_EOF\n")


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
    args = parser.parse_args()

    notebooks = collect_notebooks(args.notebooks_dir)
    print(f"Found {len(notebooks)} notebooks")

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(notebooks)

    batches = split_into_batches(notebooks, args.num_batches)
    write_github_output(batches)


if __name__ == "__main__":
    main()
