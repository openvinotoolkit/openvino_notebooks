"""Writer of the per-job dashboard fragment (fragment.json + logs/ + envs/).

Stdlib-only: runs inside the notebook test environments on Linux and Windows.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from nightly_report.constants import (
    ENVS_DIR_NAME,
    FRAGMENT_FILE_NAME,
    KEY_PACKAGES,
    LOG_HEAD_BYTES,
    LOG_MAX_BYTES,
    LOGS_DIR_NAME,
    SCHEMA_VERSION,
    Status,
    notebook_slug,
)

_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(.*)$")
_JOB_MANAGED_KEYS = ("started_at", "updated_at", "finished_at", "complete")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pip_freeze(text: str) -> dict[str, str]:
    """Parse `pip freeze` output into normalized package name -> version (or '@ <url>' / verbatim specifier)."""
    packages: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e ") or line.startswith("-e\t"):
            location = line[2:].strip()
            if "#egg=" not in location:
                continue
            egg = location.split("#egg=", 1)[1].split("&", 1)[0].strip()
            if egg:
                packages[normalize_package_name(egg)] = f"-e {location}"
            continue
        if line.startswith("-"):
            continue
        match = _REQUIREMENT_RE.match(line)
        if not match:
            continue
        name, rest = match.group(1), match.group(2).strip()
        if rest.startswith("==") and not rest.startswith("==="):
            value = rest[2:].strip()
        elif rest.startswith("@"):
            value = f"@ {rest[1:].strip()}"
        else:
            value = rest
        packages[normalize_package_name(name)] = value
    return packages


def key_packages(freeze: dict[str, str]) -> dict[str, str]:
    return {name: freeze[name] for name in KEY_PACKAGES if name in freeze}


def env_id_for(freeze: dict[str, str]) -> str:
    content = "\n".join(sorted(f"{name}=={version}" for name, version in freeze.items()))
    return hashlib.sha1(content.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def cap_log_file(path: Path, max_bytes: int = LOG_MAX_BYTES, head_bytes: int = LOG_HEAD_BYTES) -> None:
    """Truncate the middle of a log larger than max_bytes, keeping its head and tail (binary-safe, in place)."""
    path = Path(path)
    size = path.stat().st_size
    if size <= max_bytes:
        return
    head_bytes = max(0, min(head_bytes, max_bytes))
    tail_bytes = max_bytes - head_bytes
    with path.open("rb") as f:
        head = f.read(head_bytes)
        f.seek(size - tail_bytes)
        tail = f.read(tail_bytes)
    marker = f"\n...[truncated {size - head_bytes - tail_bytes} bytes]...\n".encode("utf-8")
    _atomic_write_bytes(path, head + marker + tail)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        # os.replace can transiently fail on Windows while another process (e.g. antivirus) holds the target open.
        for attempt in range(5):
            try:
                os.replace(tmp_name, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _clean_version(value) -> Optional[str]:
    """Versions reported as missing ('OpenVINO is missing', 'N/A', ...) -> None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A" or "missing" in text.lower():
        return None
    return text


def _posix(notebook: Union[str, Path]) -> str:
    if isinstance(notebook, Path):
        return notebook.as_posix()
    return str(notebook).replace("\\", "/")


class FragmentWriter:
    """Maintains fragment.json of one test job; the file is rewritten after every change so interrupted jobs keep partial results."""

    def __init__(self, out_dir: Path, job: dict, planned: list[str]) -> None:
        self.out_dir = Path(out_dir).absolute()
        self.logs_dir = self.out_dir / LOGS_DIR_NAME
        self.envs_dir = self.out_dir / ENVS_DIR_NAME
        for directory in (self.out_dir, self.logs_dir, self.envs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / FRAGMENT_FILE_NAME

        job_data = {key: value for key, value in job.items() if key not in _JOB_MANAGED_KEYS}
        if "openvino_base_version" in job_data:
            job_data["openvino_base_version"] = _clean_version(job_data["openvino_base_version"])
        now = utc_now_iso()
        job_data.update(started_at=now, updated_at=now, finished_at=None, complete=False)
        self.data = {
            "schema_version": SCHEMA_VERSION,
            "job": job_data,
            "planned": [_posix(notebook) for notebook in planned],
            "results": [],
        }
        self.save()

    def log_path(self, notebook: str) -> Path:
        return self.logs_dir / f"{notebook_slug(_posix(notebook))}.log"

    def add_result(
        self,
        notebook: str,
        status: str,
        *,
        skip_reason: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        duration_s: Optional[float] = None,
        return_code: Optional[int] = None,
        peak_rss_bytes: Optional[int] = None,
        openvino_before: Optional[str] = None,
        openvino_after: Optional[str] = None,
        env_freeze_file: Optional[Path] = None,
        error_hint: Optional[str] = None,
    ) -> dict:
        notebook = _posix(notebook)

        packages: dict[str, str] = {}
        env_id = None
        if env_freeze_file is not None and Path(env_freeze_file).is_file():
            freeze = parse_pip_freeze(Path(env_freeze_file).read_text(encoding="utf-8", errors="replace"))
            if freeze:
                packages = key_packages(freeze)
                env_id = env_id_for(freeze)
                env_copy = self.envs_dir / f"{env_id}.txt"
                if not env_copy.exists():
                    shutil.copyfile(env_freeze_file, env_copy)

        log_file = None
        log_path = self.log_path(notebook)
        if log_path.is_file():
            if status == Status.PASSED:
                log_path.unlink()
            else:
                cap_log_file(log_path)
                log_file = log_path.relative_to(self.out_dir).as_posix()

        result = {
            "notebook": notebook,
            "status": status,
            "skip_reason": skip_reason,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_s": float(duration_s) if duration_s is not None else None,
            "return_code": int(return_code) if return_code is not None else None,
            "peak_rss_bytes": int(peak_rss_bytes) if peak_rss_bytes is not None else None,
            "openvino_before": _clean_version(openvino_before),
            "openvino_after": _clean_version(openvino_after),
            "packages": packages,
            "env_id": env_id,
            "log_file": log_file,
            "error_hint": error_hint,
        }

        results = self.data["results"]
        for index, existing in enumerate(results):
            if existing["notebook"] == notebook:
                results[index] = result
                break
        else:
            results.append(result)
        self.save()
        return result

    def finish(self) -> None:
        self.data["job"]["complete"] = True
        self.data["job"]["finished_at"] = utc_now_iso()
        self.save()

    def save(self) -> None:
        self.data["job"]["updated_at"] = utc_now_iso()
        content = json.dumps(self.data, indent=1, ensure_ascii=False) + "\n"
        _atomic_write_bytes(self.path, content.encode("utf-8"))
