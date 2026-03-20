#!/usr/bin/env python3
"""
Script to collect metadata for archived (removed) notebooks from release branches.

Usage:
    python selector/src/notebook-metadata/collect-archived-notebooks.py

Prerequisites:
    - Python >= 3.10 (uses `X | Y` union type syntax)
    - upstream remote must point to openvinotoolkit/openvino_notebooks
    - run `git fetch upstream` before running this script

Output:
    selector/src/notebook-metadata/archived-notebooks.json
"""

import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "archived-notebooks.json")

REMOTE = os.environ.get("REMOTE", "upstream")


def discover_release_branches() -> list[str]:
    """Auto-discover release branches matching pattern YYYY.N from the remote."""
    result = subprocess.run(
        ["git", "branch", "-r", "--list", f"{REMOTE}/*"],
        capture_output=True,
        check=True,
    )
    output = result.stdout.decode("utf-8", errors="replace")
    pattern = re.compile(rf"^\s*{re.escape(REMOTE)}/(\d{{4}}\.\d+)\s*$", re.MULTILINE)
    branches = pattern.findall(output)
    # Sort descending: newest first
    branches.sort(key=lambda b: _branch_sort_key(b), reverse=True)
    return branches


# Pattern: digits followed by dash at the start of a path segment
_NUMERIC_PREFIX_RE = re.compile(r"(?<=/)(\d+-)")


def normalize_notebook_path(path: str) -> str:
    """Normalize notebook path by stripping numeric prefixes from all segments.

    E.g. 'notebooks/001-hello-world/001-hello-world.ipynb' -> 'notebooks/hello-world/hello-world.ipynb'
    """
    return _NUMERIC_PREFIX_RE.sub("", path)


def list_notebooks(ref: str) -> list[str]:
    """List all .ipynb files in notebooks/ directory for a given git ref."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", "notebooks/"],
        capture_output=True,
        check=True,
    )
    output = result.stdout.decode("utf-8", errors="replace")
    return [f for f in output.strip().split("\n") if f.endswith(".ipynb") and ".ipynb_checkpoints" not in f and "notebook_utils.ipynb" not in f]


def get_notebook_json(ref: str, file_path: str) -> dict | None:
    """Read and parse a notebook JSON from a given git ref."""
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{file_path}"],
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout.decode("utf-8", errors="replace"))
    except (subprocess.CalledProcessError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  Warning: Could not read {file_path} from {ref}: {e}", file=sys.stderr)
        return None


def extract_title(notebook_json: dict) -> str:
    """Extract notebook title from first markdown cell."""
    cells = notebook_json.get("cells", [])
    if not cells:
        return ""
    first_cell_content = "".join(cells[0].get("source", []))
    match = re.search(r"# (.+)", first_cell_content)
    if not match:
        return ""
    title = match.group(1).strip()
    # Remove markdown links: [text](url) -> text
    title = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", title)
    return title.strip()


def extract_openvino_metadata(notebook_json: dict) -> dict:
    """Extract imageUrl and tags from openvino_notebooks metadata."""
    ov_meta = notebook_json.get("metadata", {}).get("openvino_notebooks", {})
    return {
        "imageUrl": ov_meta.get("imageUrl") or None,
        "tags": ov_meta.get("tags", {"categories": [], "tasks": [], "libraries": [], "other": []}),
    }


def main():
    release_branches = discover_release_branches()
    if not release_branches:
        print(f"No release branches found for remote '{REMOTE}'. Run 'git fetch {REMOTE}' first.", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(release_branches)} release branches: {', '.join(release_branches)}")
    print("Collecting list of notebooks in latest branch...")
    latest_ref = f"{REMOTE}/latest"
    latest_notebooks_raw = list_notebooks(latest_ref)
    # Build set of both raw and normalized paths for matching
    latest_notebooks: set[str] = set()
    for nb in latest_notebooks_raw:
        latest_notebooks.add(nb)
        latest_notebooks.add(normalize_notebook_path(nb))
    print(f"  Found {len(latest_notebooks_raw)} notebooks in latest")

    archived_map: dict[str, dict] = {}

    for branch in release_branches:
        ref = f"{REMOTE}/{branch}"
        print(f"\nScanning branch {branch}...")

        try:
            branch_notebooks = list_notebooks(ref)
        except subprocess.CalledProcessError:
            print(f"  Branch {branch} not found, skipping")
            continue

        print(f"  Found {len(branch_notebooks)} notebooks")
        added = 0

        for nb_path in branch_notebooks:
            normalized = normalize_notebook_path(nb_path)

            # Skip if still exists in latest (check both raw and normalized path)
            if nb_path in latest_notebooks or normalized in latest_notebooks:
                continue

            # Skip if already found in a newer branch (check both raw and normalized)
            if nb_path in archived_map or normalized in archived_map:
                continue

            notebook_json = get_notebook_json(ref, nb_path)
            if notebook_json is None:
                continue

            title = extract_title(notebook_json)
            meta = extract_openvino_metadata(notebook_json)

            # Path relative to notebooks/ dir
            relative_path = nb_path.removeprefix("notebooks/")

            # Use normalized path as key to prevent duplicates across renamed branches
            archived_map[normalized] = {
                "title": title or relative_path,
                "path": relative_path,
                "imageUrl": meta["imageUrl"],
                "lastBranch": branch,
                "githubUrl": f"https://github.com/openvinotoolkit/openvino_notebooks/blob/{branch}/{nb_path}",
                "tags": meta["tags"],
            }
            added += 1

        print(f"  Added {added} archived notebooks from {branch}")

    # Sort by lastBranch (newest first), then by title
    archived = sorted(
        archived_map.values(),
        key=lambda x: (-_branch_sort_key(x["lastBranch"]), x["title"].lower()),
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(archived, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Wrote {len(archived)} archived notebooks to {OUTPUT_FILE}")


def _branch_sort_key(branch: str) -> float:
    """Convert branch name like '2025.4' to sortable numeric value."""
    parts = branch.split(".")
    try:
        return float(parts[0]) * 100 + float(parts[1])
    except (IndexError, ValueError):
        return 0


if __name__ == "__main__":
    main()
