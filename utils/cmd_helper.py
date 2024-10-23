import os
import subprocess  # nosec - disable B404:import-subprocess check
import sys
from pathlib import Path


def clone_repo(repo_url: str, revision: str = None, add_to_sys_path: bool = True):
    repo_path = Path(repo_url.split("/")[-1].replace(".git", ""))

    if not repo_path.exists():
        exit_code = os.system(f"git clone {repo_url}")
        if exit_code != 0:
            raise Exception("Failed to clone the repository!")
        if revision:
            subprocess.Popen(["git", "checkout", revision], cwd=str(repo_path))
    if add_to_sys_path:
        sys.path.insert(0, str(repo_path))
