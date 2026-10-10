"""Shared constants of the nightly dashboard data contract.

Stdlib-only: imported from test jobs (validate_notebooks.py) and from the aggregation job.
"""

SCHEMA_VERSION = "1.0.0"

FRAGMENT_FILE_NAME = "fragment.json"
PLAN_FILE_NAME = "plan.json"
REPORT_FILE_NAME = "nightly-report.json"
ENVIRONMENTS_FILE_NAME = "environments.json"
SUMMARY_FILE_NAME = "summary.md"

LOGS_DIR_NAME = "logs"
ENVS_DIR_NAME = "envs"

FRAGMENT_ARTIFACT_PREFIX = "nightly-fragment-"
PLAN_ARTIFACT_NAME = "nightly-plan"
REPORT_ARTIFACT_NAME = "nightly-dashboard-data"


class Status:
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"
    NOT_RUN = "not_run"

    ALL = (PASSED, FAILED, TIMEOUT, ERROR, SKIPPED, NOT_RUN)
    EXECUTED = (PASSED, FAILED, TIMEOUT, ERROR)
    FAILING = (FAILED, TIMEOUT, ERROR)


class SkipReason:
    SKIP_CONFIG = "skip_config"
    IGNORE_LIST = "ignore_list"

    ALL = (SKIP_CONFIG, IGNORE_LIST)


class NotRunReason:
    # Followed by ":<failed step name>", e.g. "job_failed_before_tests:Install python dependencies"
    JOB_FAILED_BEFORE_TESTS = "job_failed_before_tests"
    JOB_CANCELLED = "job_cancelled"
    JOB_SKIPPED = "job_skipped"
    JOB_INTERRUPTED = "job_interrupted"
    NO_FRAGMENT = "no_fragment"


class FailureCategory:
    CELL_ERROR = "cell_error"
    KERNEL_DIED = "kernel_died"
    TIMEOUT = "timeout"
    HARNESS_ERROR = "harness_error"
    UNKNOWN = "unknown"

    ALL = (CELL_ERROR, KERNEL_DIED, TIMEOUT, HARNESS_ERROR, UNKNOWN)


class ErrorClass:
    """Heuristic triage class of a failure; see error_parser.classify()."""

    TIMEOUT = "timeout"
    HARNESS = "harness"
    DISK = "disk"
    MEMORY = "memory"
    CRASH = "crash"
    DEPENDENCY = "dependency"
    NETWORK = "network"
    OPENVINO = "openvino"
    ASSERTION = "assertion"
    OTHER = "other"

    ALL = (TIMEOUT, HARNESS, DISK, MEMORY, CRASH, DEPENDENCY, NETWORK, OPENVINO, ASSERTION, OTHER)


# Packages whose versions are stored inline in every result (normalized pip names).
KEY_PACKAGES = (
    "openvino",
    "openvino-genai",
    "openvino-tokenizers",
    "nncf",
    "optimum",
    "optimum-intel",
    "transformers",
    "torch",
    "torchvision",
    "diffusers",
    "huggingface-hub",
    "numpy",
)

# Size caps for data stored in fragments and reports.
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_HEAD_BYTES = 256 * 1024
EXCERPT_MAX_BYTES = 8 * 1024
TRACEBACK_MAX_LINES = 60
LOG_TAIL_MAX_LINES = 100
CELL_SOURCE_MAX_LINES = 40
MESSAGE_MAX_CHARS = 1024

NEAR_TIMEOUT_RATIO = 0.8


def config_id(device: str, os_name: str, python: str) -> str:
    return f"{device}-{os_name}-py{python}"


def notebook_slug(notebook: str) -> str:
    """'async-api/async-api.ipynb' -> 'async-api__async-api'."""
    name = notebook.replace("\\", "/")
    if name.endswith(".ipynb"):
        name = name[: -len(".ipynb")]
    return name.replace("/", "__")
