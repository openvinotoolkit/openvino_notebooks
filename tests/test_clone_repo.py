import subprocess
from unittest.mock import patch

import pytest

from utils.cmd_helper import clone_repo

REPO_URL = "https://github.com/example/some-repo.git"


def test_clone_repo_reports_missing_git(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with patch("subprocess.run", side_effect=FileNotFoundError(2, "No such file or directory: 'git'")):
        with pytest.raises(FileNotFoundError):
            clone_repo(REPO_URL, add_to_sys_path=False)
    assert "git was not found" in capsys.readouterr().out


def test_clone_repo_prints_git_error_when_clone_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    error = subprocess.CalledProcessError(128, ["git", "clone"], stderr="fatal: repository not found")
    with patch("subprocess.run", side_effect=error):
        with pytest.raises(subprocess.CalledProcessError):
            clone_repo(REPO_URL, add_to_sys_path=False)
    assert "fatal: repository not found" in capsys.readouterr().out


def test_clone_repo_does_not_hide_other_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("subprocess.run", side_effect=PermissionError("access denied")):
        with pytest.raises(PermissionError):
            clone_repo(REPO_URL, add_to_sys_path=False)


def test_clone_repo_checks_out_revision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("subprocess.run") as run:
        repo_path = clone_repo(REPO_URL, revision="abc123", add_to_sys_path=False)
    assert str(repo_path) == "some-repo"
    assert run.call_args_list[0][0][0] == ["git", "clone", REPO_URL]
    assert run.call_args_list[1][0][0] == ["git", "checkout", "abc123"]
