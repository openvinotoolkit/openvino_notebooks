import subprocess

import pytest

from utils.cmd_helper import clone_repo


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def local_repository(tmp_path, monkeypatch):
    source = tmp_path / "source" / "sample-repo"
    source.mkdir(parents=True)
    git(source, "init", "--quiet")
    for version in ("first", "second"):
        (source / "version.txt").write_text(version)
        git(source, "add", "version.txt")
        git(source, "-c", "user.name=Test Contributor", "-c", "user.email=test@example.com", "commit", "--quiet", "--no-gpg-sign", "-m", version)
        git(source, "tag", version)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return source


@pytest.mark.parametrize("revision", ["first", "commit"])
def test_clone_repo_checks_out_revision_in_existing_repository(local_repository, revision):
    repo = clone_repo(str(local_repository), add_to_sys_path=False)
    assert (repo / "version.txt").read_text() == "second"
    requested = git(local_repository, "rev-parse", "first") if revision == "commit" else revision

    result = clone_repo(str(local_repository), revision=requested, add_to_sys_path=False)

    assert result == repo
    assert git(result, "rev-parse", "HEAD") == git(local_repository, "rev-parse", "first")
    assert (result / "version.txt").read_text() == "first"


def test_clone_repo_can_change_revision_between_calls(local_repository):
    clone_repo(str(local_repository), revision="first", add_to_sys_path=False)

    repo = clone_repo(str(local_repository), revision="second", add_to_sys_path=False)

    assert (repo / "version.txt").read_text() == "second"


def test_clone_repo_rejects_missing_revision_in_existing_repository(local_repository):
    repo = clone_repo(str(local_repository), add_to_sys_path=False)
    original_head = git(repo, "rev-parse", "HEAD")

    with pytest.raises(subprocess.CalledProcessError):
        clone_repo(str(local_repository), revision="missing-revision", add_to_sys_path=False)

    assert git(repo, "rev-parse", "HEAD") == original_head


def test_clone_repo_checks_out_revision_for_new_repository(local_repository):
    repo = clone_repo(str(local_repository), revision="first", add_to_sys_path=False)

    assert (repo / "version.txt").read_text() == "first"


def test_clone_repo_preserves_existing_revision_when_none_requested(local_repository):
    clone_repo(str(local_repository), revision="first", add_to_sys_path=False)

    repo = clone_repo(str(local_repository), add_to_sys_path=False)

    assert (repo / "version.txt").read_text() == "first"


def test_clone_repo_preserves_local_changes_when_checkout_fails(local_repository):
    repo = clone_repo(str(local_repository), add_to_sys_path=False)
    (repo / "version.txt").write_text("local work")

    with pytest.raises(subprocess.CalledProcessError):
        clone_repo(str(local_repository), revision="first", add_to_sys_path=False)

    assert (repo / "version.txt").read_text() == "local work"
