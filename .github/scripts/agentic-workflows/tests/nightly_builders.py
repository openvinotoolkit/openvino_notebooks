"""Builders of synthetic nightly reports and an in-memory stand-in for the `gh` CLI, shared by the collector tests."""

import io
import json
import re
import zipfile

import collector_common as common
import collector_analysis as analysis
import collector_evidence as evidence

REPO = "acme/notebooks"
NIGHTLY_ID = 1001

CPU_CONFIG = "cpu-ubuntu-22.04-py3.11"
CPU_CONFIG_312 = "cpu-ubuntu-22.04-py3.12"
CPU_CONFIG_313 = "cpu-ubuntu-22.04-py3.13"
GPU_CONFIG = "dgpu-ubuntu-22.04-py3.11"

SIG_DEPENDENCY = "aaaaaaaaaaaa"
SIG_OTHER = "bbbbbbbbbbbb"
SIG_TIMEOUT = "cccccccccccc"
INFRA_STEP = "GPU diagnostics (OpenVINO devices)"
INFRA_REASON = f"job_failed_before_tests:{INFRA_STEP}"
SIG_INFRA = analysis.infra_signature(INFRA_REASON)

ENVIRONMENTS = {
    "env-fail": {"openvino": "2026.1.0", "protobuf": "4.25.0", "numpy": "2.0.0", "pillow": "11.0.0"},
    "env-pass": {"openvino": "2026.0.0", "protobuf": "6.31.0", "numpy": "2.0.0", "pillow": "11.0.0"},
}

JOB_LOG = "\n".join(
    [f"2026-01-10T00:00:{i:02d}.0000000Z \x1b[36mline {i}\x1b[0m" for i in range(1, 40)]
    + ["2026-01-10T00:01:00.0000000Z \x1b[31m##[error]Process completed with exit code 1.\x1b[0m"]
    + [f"2026-01-10T00:02:{i:02d}.0000000Z after {i}" for i in range(1, 10)]
)


def make_run(run_id=NIGHTLY_ID, number=10, created_at="2026-01-10T00:00:00Z", conclusion="failure"):
    return {
        "id": run_id,
        "number": number,
        "workflow": "treon_nightly",
        "repository": REPO,
        "event": "schedule",
        "branch": "latest",
        "nightly_date": created_at[:10],
        "created_at": created_at,
        "conclusion": conclusion,
        "url": f"https://github.com/{REPO}/actions/runs/{run_id}",
        "commit": {"short_sha": "abc1234", "message": "Bump something (#1)"},
        "inputs": {"ov_branch": "master", "docker_tag": "pr-1"},
    }


def make_failure(notebook, config_id, signature, error_class="other", status="failed", message="boom", exception_type="ValueError", job_id=11, artifact_id=77):
    return {
        "id": f"{config_id}/{notebook}",
        "notebook": notebook,
        "config_id": config_id,
        "job_id": job_id,
        "status": status,
        "category": "timeout" if status == "timeout" else "cell_error",
        "error_class": error_class,
        "signature": signature,
        "cell_index": 3,
        "cell_source_excerpt": "import foo\nfoo.bar()",
        "exception_type": exception_type,
        "exception_message": message,
        "traceback_excerpt": f"Traceback (most recent call last)\n{message}",
        "log_excerpt": "Start executing cell 3",
        "log_excerpt_source": "job_log",
        "log_excerpt_lines": {"start": 1, "end": 5, "error_line": 4, "test_start": 1},
        "links": {
            "job": f"https://github.com/{REPO}/actions/runs/{NIGHTLY_ID}/job/{job_id}",
            "logs_artifact": f"https://github.com/{REPO}/actions/runs/{NIGHTLY_ID}/artifacts/{artifact_id}",
            "log_file": f"logs/{evidence.safe_name(notebook)}__{config_id}.log",
        },
    }


def make_result(notebook, config_id, status="passed", job_id=11, not_run_reason=None, env_id=None, openvino=None):
    return {
        "notebook": notebook,
        "config_id": config_id,
        "job_id": job_id,
        "status": status,
        "not_run_reason": not_run_reason,
        "packages": {"openvino": openvino} if openvino else {},
        "env_id": env_id,
    }


def make_job(job_id, config_id, step=None):
    return {
        "id": job_id,
        "name": f"job {job_id}",
        "config_id": config_id,
        "batch": "batch_0",
        "conclusion": "failure",
        "runner_name": "runner-1",
        "url": f"https://github.com/{REPO}/actions/runs/{NIGHTLY_ID}/job/{job_id}",
        "failed_step": {"number": 14, "name": step, "url": f"https://github.com/{REPO}/actions/runs/{NIGHTLY_ID}/job/{job_id}#step:14:1"} if step else None,
    }


def make_report(failures=(), results=(), run=None, jobs=(), error_groups=(), schema_version="1.2.0"):
    configs = [CPU_CONFIG, CPU_CONFIG_312, CPU_CONFIG_313, GPU_CONFIG]
    return {
        "schema_version": schema_version,
        "generated_at": "2026-01-10T04:00:00Z",
        "run": run or make_run(),
        "configs": [{"id": c, "os_family": "linux", "python": c.rsplit("py", 1)[1], "device": "gpu" if c.startswith("dgpu") else "cpu"} for c in configs],
        "jobs": list(jobs),
        "notebooks": [],
        "results": list(results),
        "failures": list(failures),
        "error_groups": list(error_groups),
        "summary": {"infra": {"jobs_failed_before_tests": 1, "infra_failure_rate": 0.1}},
    }


def make_zip(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def nightly_report():
    """Two PR-candidate groups, a timeout group and an infra group; one notebook (a) fails twice and also passes on a third config."""
    dependency = "ImportError: cannot import name 'runtime_version' from 'google.protobuf'"
    failures = [
        make_failure("a/a.ipynb", CPU_CONFIG, SIG_DEPENDENCY, "dependency", message=dependency, exception_type="ImportError"),
        make_failure("a/a.ipynb", CPU_CONFIG_312, SIG_DEPENDENCY, "dependency", message=dependency, exception_type="ImportError"),
        make_failure("b/b.ipynb", CPU_CONFIG, SIG_OTHER, "other", message="b.txt not found", exception_type="FileNotFoundError"),
        make_failure("c/c.ipynb", CPU_CONFIG, SIG_TIMEOUT, "timeout", status="timeout", message="Timeout reached (3600s), process killed", exception_type=None),
    ]
    results = [
        make_result("a/a.ipynb", CPU_CONFIG, "failed", env_id="env-fail", openvino="2026.1.0"),
        make_result("a/a.ipynb", CPU_CONFIG_312, "failed", env_id="env-fail", openvino="2026.1.0"),
        make_result("a/a.ipynb", CPU_CONFIG_313, "passed", env_id="env-pass", openvino="2026.0.0"),
        make_result("b/b.ipynb", CPU_CONFIG, "failed"),
        make_result("c/c.ipynb", CPU_CONFIG, "timeout"),
        make_result("d/d.ipynb", GPU_CONFIG, "not_run", job_id=13, not_run_reason=INFRA_REASON),
        make_result("e/e.ipynb", GPU_CONFIG, "not_run", job_id=13, not_run_reason=INFRA_REASON),
        make_result("f/f.ipynb", GPU_CONFIG, "not_run", job_id=13, not_run_reason="job_cancelled"),
        make_result("g/g.ipynb", CPU_CONFIG, "passed"),
    ]
    jobs = [make_job(11, CPU_CONFIG), make_job(12, CPU_CONFIG_312), make_job(13, GPU_CONFIG, step=INFRA_STEP)]
    return make_report(failures, results, jobs=jobs)


def previous_nights():
    """Night 1000 had neither dependency failure; night 999 had all three error groups."""
    night_1000 = make_report(
        run=make_run(1000, 9, "2026-01-09T00:00:00Z"),
        failures=[
            make_failure("b/b.ipynb", CPU_CONFIG, SIG_OTHER, "other"),
            make_failure("c/c.ipynb", CPU_CONFIG, SIG_TIMEOUT, "timeout", status="timeout"),
        ],
        results=[
            make_result("a/a.ipynb", CPU_CONFIG, "passed"),
            make_result("a/a.ipynb", CPU_CONFIG_312, "passed"),
            make_result("b/b.ipynb", CPU_CONFIG, "failed"),
            make_result("c/c.ipynb", CPU_CONFIG, "timeout"),
        ],
    )
    night_999 = make_report(
        run=make_run(999, 8, "2026-01-08T00:00:00Z"),
        failures=[
            make_failure("a/a.ipynb", CPU_CONFIG, SIG_DEPENDENCY, "dependency"),
            make_failure("b/b.ipynb", CPU_CONFIG, SIG_OTHER, "other"),
            make_failure("c/c.ipynb", CPU_CONFIG, SIG_TIMEOUT, "timeout", status="timeout"),
        ],
        results=[
            make_result("a/a.ipynb", CPU_CONFIG, "failed"),
            make_result("a/a.ipynb", CPU_CONFIG_312, "passed"),
            make_result("b/b.ipynb", CPU_CONFIG, "failed"),
            make_result("c/c.ipynb", CPU_CONFIG, "timeout"),
        ],
    )
    return night_1000, night_999


class FakeGh:
    """Stands in for `github.gh`: answers the GitHub API and `gh pr|issue list` calls from memory and records them."""

    def __init__(self):
        self.calls = []
        self.runs = {}
        self.artifacts = {}
        self.zips = {}
        self.job_logs = {}
        self.prs = []
        self.issues = []
        self.issues_disabled = False
        self.fail_on = None

    def add_night(self, report, artifact_id, environments=None):
        run = report["run"]
        self.runs[run["id"]] = {
            "id": run["id"],
            "path": ".github/workflows/treon_nightly.yml",
            "status": "completed",
            "conclusion": run["conclusion"],
            "event": "schedule",
            "created_at": run["created_at"],
        }
        self.artifacts[run["id"]] = [{"id": artifact_id, "name": common.REPORT_ARTIFACT, "expired": False}]
        members = {common.REPORT_FILE: json.dumps(report)}
        if environments is not None:
            members[common.ENVIRONMENTS_FILE] = json.dumps(environments)
        self.zips[artifact_id] = make_zip(members)

    def add_fragment(self, artifact_id, report):
        self.zips[artifact_id] = make_zip({f["links"]["log_file"]: f"\x1b[31mstart\x1b[0m\r\nlog of {f['id']}\r\n" for f in report["failures"]})

    def count(self, text):
        return sum(1 for call in self.calls if text in " ".join(call))

    def __call__(self, args, stdout_path=None, retries=3, timeout=600):
        self.calls.append(list(args))
        if self.fail_on and self.fail_on in " ".join(args):
            raise common.GhError(args, 1, "HTTP 500: server error")
        if args[:2] == ["pr", "list"]:
            return json.dumps(self.prs).encode()
        if args[:2] == ["issue", "list"]:
            if self.issues_disabled:
                raise common.GhError(args, 1, f"the '{REPO}' repository has disabled issues")
            return json.dumps(self.issues).encode()
        assert args[0] == "api" and args[1].startswith(f"repos/{REPO}/"), args
        path = args[1].split("?")[0][len(f"repos/{REPO}/") :]
        if path == "actions/workflows/treon_nightly.yml/runs":
            runs = [r for r in self.runs.values() if r["event"] == "schedule" and r["status"] == "completed"]
            return json.dumps({"workflow_runs": sorted(runs, key=lambda r: r["created_at"], reverse=True)}).encode()
        match = re.fullmatch(r"actions/runs/(\d+)", path)
        if match:
            return json.dumps(self.runs[int(match.group(1))]).encode()
        match = re.fullmatch(r"actions/runs/(\d+)/artifacts", path)
        if match:
            return json.dumps({"artifacts": self.artifacts.get(int(match.group(1)), [])}).encode()
        match = re.fullmatch(r"actions/artifacts/(\d+)/zip", path)
        if match:
            stdout_path.write_bytes(self.zips[int(match.group(1))])
            return b""
        match = re.fullmatch(r"actions/jobs/(\d+)/logs", path)
        if match:
            if int(match.group(1)) not in self.job_logs:
                raise common.GhError(args, 1, "HTTP 404: Not Found")
            return self.job_logs[int(match.group(1))].encode()
        raise AssertionError(f"unexpected gh call: {args}")


def nightly_world(fake):
    """Registers the current nightly (run 1001, report artifact 55, fragment artifact 77, job log 13) and two earlier nights."""
    report = nightly_report()
    fake.add_night(report, 55, ENVIRONMENTS)
    fake.add_fragment(77, report)
    fake.job_logs[13] = JOB_LOG
    night_1000, night_999 = previous_nights()
    fake.add_night(night_1000, 501)
    fake.add_night(night_999, 502)
    return report
