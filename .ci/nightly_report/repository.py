"""Data from the repository checkout: git metadata and the notebook catalog."""

import json
import re
import subprocess  # nosec B404 - runs git with fixed arguments
from pathlib import Path
from typing import Iterable, Optional

from nightly_report.report_common import Links, Warnings, format_time, parse_time


def run_git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), *args], stderr=subprocess.DEVNULL, text=True, encoding="utf-8")  # nosec B603 B607
    except (OSError, subprocess.CalledProcessError):
        return None


def get_tree_shas(repo_root: Path, notebooks_dir: str) -> dict[str, str]:
    output = run_git(repo_root, "ls-tree", "-r", "-d", "HEAD", "--", f"{notebooks_dir}/")
    tree_shas: dict[str, str] = {}
    for line in (output or "").splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "tree" and path.startswith(f"{notebooks_dir}/"):
            tree_shas[path[len(notebooks_dir) + 1 :]] = parts[2]
    return tree_shas


def get_last_modified(repo_root: Path, notebooks_dir: str, dirs: Iterable[str], warnings: Warnings) -> dict[str, dict]:
    """Latest commit touching each notebook directory, found in a single pass over the history."""
    pending = set(dirs)
    if not pending:
        return {}
    if (run_git(repo_root, "rev-parse", "--is-shallow-repository") or "").strip() != "false":
        warnings.add("Repository checkout is shallow or not a git repository; notebook last_modified is not collected.")
        return {}
    command = ["git", "-C", str(repo_root), "log", "--format=%x1e%H%x1f%cI%x1f%an%x1f%s", "--name-only", "HEAD", "--", f"{notebooks_dir}/"]
    found: dict[str, dict] = {}
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")  # nosec B603
    except OSError as error:
        warnings.add(f"git log failed: {error}")
        return {}
    commit: Optional[dict] = None
    try:
        for line in process.stdout:
            line = line.rstrip("\n")
            if line.startswith("\x1e"):
                sha, date, author, subject = (line[1:].split("\x1f") + ["", "", "", ""])[:4]
                commit = {"sha": sha, "date": date, "author": author, "message": subject}
                continue
            if not line or commit is None or not line.startswith(f"{notebooks_dir}/"):
                continue
            parts = line[len(notebooks_dir) + 1 :].split("/")[:-1]
            for depth in range(1, len(parts) + 1):
                directory = "/".join(parts[:depth])
                if directory in pending:
                    pending.discard(directory)
                    found[directory] = commit
            if not pending:
                break
    finally:
        process.kill()
        process.wait()
    return found


def get_head_commit(repo_root: Path) -> Optional[dict]:
    output = run_git(repo_root, "log", "-1", "--format=%H%x1f%cI%x1f%an%x1f%s")
    if not output:
        return None
    sha, date, author, subject = (output.strip().split("\x1f") + ["", "", "", ""])[:4]
    return {"sha": sha, "date": date, "author": author, "message": subject}


def read_notebook_info(path: Path) -> tuple[str, dict]:
    title, tags = "", {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return title, tags
    cells = content.get("cells") or []
    if cells:
        source = cells[0].get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        match = re.search(r"# (.+)", source)
        if match:
            title = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", match.group(1)).strip()
    raw_tags = (content.get("metadata") or {}).get("openvino_notebooks", {}).get("tags") or {}
    if isinstance(raw_tags, dict):
        tags = {key: [str(item) for item in value] for key, value in raw_tags.items() if isinstance(value, list)}
    return title, tags


def commit_entry(commit: Optional[dict], links: Links) -> Optional[dict]:
    if not commit or not commit.get("sha"):
        return None
    sha = commit["sha"]
    return {
        "sha": sha,
        "short_sha": sha[:7],
        "message": (commit.get("message") or "").splitlines()[0] if commit.get("message") else None,
        "author": commit.get("author") or None,
        "date": format_time(parse_time(commit.get("date"))),
        "url": links.commit(sha),
    }


def collect_notebooks(
    repo_root: Path, notebooks_dir: str, head_sha: str, latest_ref: Optional[str], links: Links, results: list[dict], warnings: Warnings
) -> list[dict]:
    """Catalog of the notebooks at the tested commit, including notebooks with results that are missing in the checkout."""
    notebooks_root = repo_root / notebooks_dir
    paths = set()
    if notebooks_root.exists():
        for path in notebooks_root.rglob("*.ipynb"):
            relative = path.relative_to(notebooks_root)
            if path.name.startswith("test_") or ".ipynb_checkpoints" in relative.parts:
                continue
            paths.add(relative.as_posix())
    else:
        warnings.add(f"Notebooks directory '{notebooks_root}' does not exist.")
    missing = {row["notebook"] for row in results} - paths
    if missing:
        warnings.add(f"{len(missing)} notebook(s) with results are missing in the checkout: {sorted(missing)[:10]}")
    paths |= missing

    dirs = {str(Path(path).parent.as_posix()) for path in paths}
    tree_shas = get_tree_shas(repo_root, notebooks_dir)
    last_modified = get_last_modified(repo_root, notebooks_dir, dirs, warnings)
    catalog = []
    for path in sorted(paths):
        directory = Path(path).parent.as_posix()
        title, tags = read_notebook_info(notebooks_root / path)
        catalog.append(
            {
                "path": path,
                "dir": directory,
                "title": title or Path(path).stem,
                "tags": tags,
                "dir_tree_sha": tree_shas.get(directory),
                "last_modified": commit_entry(last_modified.get(directory), links),
                "links": {
                    "source": links.blob(head_sha, f"{notebooks_dir}/{path}"),
                    "latest": links.blob(latest_ref, f"{notebooks_dir}/{path}") if latest_ref else None,
                    "history": links.history(head_sha, f"{notebooks_dir}/{directory}"),
                },
            }
        )
    return catalog
