import os
import sys
from pathlib import Path


def clone_repo(repo_url: str, revision: str = None):
    import subprocess  # nosec - disable B404:import-subprocess check

    repo_path = Path(repo_url.split("/")[-1].split(".")[0])

    if not repo_path.exists():
        exit_code = os.system(f"git clone {repo_url}")
        if exit_code != 0:
            raise Exception("Failed to clone the repository!")
        if revision:
            subprocess.Popen(["git", "checkout", revision], cwd=str(repo_path))

    sys.path.insert(0, str(repo_path))
