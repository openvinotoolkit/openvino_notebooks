"""Reads the nightly remediation knowledge base from its repository branch for the digest (best effort: it only adds context)."""

from __future__ import annotations

from typing import Callable
from urllib.parse import urlencode

import collector_github as github
from collector_common import CollectorError, GroupDigest
from collector_render import repo_path
from knowledge_base import KNOWLEDGE_BRANCH, KNOWLEDGE_FILE, KnowledgeError, empty_knowledge, parse_knowledge, related_records


def fetch_knowledge(repo: str, warn: Callable[[str], None]) -> dict:
    """The stored knowledge base, or an empty one when the branch or file does not exist yet or cannot be read (with a warning)."""
    path = f"repos/{repo}/contents/{KNOWLEDGE_FILE}?{urlencode({'ref': KNOWLEDGE_BRANCH})}"
    try:
        raw = github.gh(["api", path, "-H", "Accept: application/vnd.github.raw"])
        return parse_knowledge(raw.decode("utf-8"))
    except CollectorError as error:
        if getattr(error, "returncode", None) is not None and "HTTP 404" in getattr(error, "stderr", ""):
            return empty_knowledge()
        warn(f"the knowledge base on {KNOWLEDGE_BRANCH} could not be read and is skipped: {error}")
    except (KnowledgeError, UnicodeDecodeError) as error:
        warn(f"the knowledge base on {KNOWLEDGE_BRANCH} is invalid and is skipped: {error}")
    return empty_knowledge()


def attach_knowledge(digest: GroupDigest, knowledge: dict) -> None:
    group = digest.group
    digest.knowledge = knowledge["signatures"].get(group.signature)
    # Records store the repository paths of `group.json`, not the report's notebook names.
    notebooks = [repo_path(notebook) for notebook in group.notebooks]
    digest.related_knowledge = related_records(knowledge, group.signature, group.exception_type, notebooks)
