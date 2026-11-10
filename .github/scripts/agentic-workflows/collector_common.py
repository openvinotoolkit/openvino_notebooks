"""Shared constants, exceptions, and record types for nightly failure collection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

NIGHTLY_WORKFLOW_FILE = "treon_nightly.yml"
NIGHTLY_BRANCH = "latest"
REPORT_ARTIFACT = "nightly-dashboard-data"
REPORT_FILE = "nightly-report.json"
ENVIRONMENTS_FILE = "environments.json"
SUPPORTED_SCHEMA_MAJOR = 1
TRACKING_LABEL = "nightly-remediator"
REJECTED_COOLDOWN = timedelta(days=30)
TRACKED_HISTORY_LIMIT = 300
INFRA_NOT_RUN_KINDS = ("job_failed_before_tests", "no_fragment", "job_interrupted")
PR_CANDIDATE_CLASSES = ("dependency", "assertion", "openvino", "other")
ALLOWED_SOURCE_REPOS = ("openvinotoolkit/openvino_notebooks",)
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_LOG_BYTES = 2 * 1024 * 1024
MAX_TOTAL_LOG_BYTES = 40 * 1024 * 1024
MAX_JOB_LOG_BYTES = 64 * 1024 * 1024
CORE_PACKAGES = {
    "openvino",
    "openvino-dev",
    "openvino-genai",
    "openvino-tokenizers",
    "nncf",
    "optimum",
    "optimum-intel",
    "torch",
    "transformers",
    "numpy",
    "huggingface-hub",
}
SIGNATURE_RE = re.compile(r"(?:Error signature|\bsig)\W{0,8}([0-9a-f]{12})\b", re.IGNORECASE)
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
JOB_LOG_TIMESTAMP_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z ")
ARTIFACT_ID_RE = re.compile(r"/artifacts/(\d+)")
ESCAPE_SEQUENCES_FLAG = "--allow-escape-sequences"
TRANSIENT_MARKERS = ("http 5", "http 429", "timeout", "timed out", "connection reset", "connection refused", "eof", "temporarily", "rate limit")
Key = Tuple[str, str]  # (notebook relative to notebooks/, config_id)


class CollectorError(RuntimeError):
    """Unexpected problem: the job fails and the agent does not run."""


class NoWork(Exception):
    """Expected condition after which the agent has nothing to do; reported as `has_work: false`."""


class GhError(CollectorError):
    def __init__(self, args: List[str], returncode: int, stderr: str) -> None:
        super().__init__(f"gh {' '.join(args[:2])} failed ({returncode}): {stderr.strip()[:300]}")
        self.returncode = returncode
        self.stderr = stderr


@dataclass
class Member:
    """One (notebook, config) pair that failed, or could not run, with the group's error."""

    notebook: str
    config_id: str
    status: str
    job_id: Optional[int] = None
    failure: Optional[dict] = None
    not_run_reason: Optional[str] = None

    @property
    def key(self) -> Key:
        return (self.notebook, self.config_id)


@dataclass
class Group:
    signature: str
    kind: str  # "error": parsed failure of a notebook; "infra": notebooks never run because a job broke
    category: str
    error_class: str
    exception_type: str
    exception_message: str
    members: List[Member] = field(default_factory=list)

    @property
    def notebooks(self) -> List[str]:
        return sorted({m.notebook for m in self.members})

    @property
    def configs(self) -> List[str]:
        return sorted({m.config_id for m in self.members})

    @property
    def headline(self) -> str:
        if self.kind == "infra":
            step = self.exception_message
            return f"{self.error_class}: {step}" if step else self.error_class
        return f"{self.exception_type}: {self.exception_message}" if self.exception_type else (self.exception_message or self.category)


@dataclass
class TrackedItem:
    kind: str  # "pr" or "issue"
    number: int
    url: str
    title: str
    state: str  # open, merged, closed, completed, not_planned, duplicate
    signature: str
    created_at: str
    closed_at: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{'PR' if self.kind == 'pr' else 'issue'} #{self.number}"

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "number": self.number,
            "url": self.url,
            "title": self.title,
            "state": self.state,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
        }


@dataclass
class TrackingDecision:
    skip: bool
    reason: str = ""
    blocking: Optional[TrackedItem] = None
    previous: List[TrackedItem] = field(default_factory=list)


@dataclass
class Night:
    info: dict  # `run` section of the report
    signatures: Dict[str, set]  # signature -> {(notebook, config_id)}
    statuses: Dict[Key, str]


@dataclass
class Context:
    repo: str
    source_repo: str
    run: dict
    report: dict
    environments: dict
    configs: Dict[str, dict]
    jobs: Dict[int, dict]
    results_by_key: Dict[Key, dict]
    results_by_notebook: Dict[str, List[dict]]


@dataclass
class GroupDigest:
    index: int
    group: Group
    trend: str
    decision: TrackingDecision
    history: List[dict]
    representatives: List[Member]
    raw_logs: Dict[Key, str]
    infra_jobs: List[dict]
    job_logs: Dict[int, str]
    versions: List[dict]
    packages: Dict[str, str]
    package_reference: Optional[str]
    env_diff: List[Tuple[str, str, str]]
    env_diff_total: int

    @property
    def dir_name(self) -> str:
        return f"{self.index:02d}-{self.group.signature}"
