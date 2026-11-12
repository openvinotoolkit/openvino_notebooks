"""Helpers shared by the report builder modules: warnings collection, timestamps, GitHub links and JSON loading."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class Warnings:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, message: str) -> None:
        print(f"WARNING: {message}", flush=True)
        self.items.append(message)


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def format_time(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def seconds_between(start: Optional[str], end: Optional[str]) -> Optional[float]:
    start_time, end_time = parse_time(start), parse_time(end)
    if start_time is None or end_time is None:
        return None
    return round(max((end_time - start_time).total_seconds(), 0.0), 1)


def first_line(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return text.strip().splitlines()[0] if text.strip() else None


class Links:
    def __init__(self, server_url: str, repository: str, run_id: int) -> None:
        self.base = f"{server_url.rstrip('/')}/{repository}"
        self.run_id = run_id

    def run(self) -> str:
        return f"{self.base}/actions/runs/{self.run_id}"

    def job(self, job_id: int) -> str:
        return f"{self.run()}/job/{job_id}"

    def step(self, job_url: str, number: int, line: int = 1) -> str:
        return f"{job_url}#step:{number}:{line}"

    def artifact(self, artifact_id: int) -> str:
        return f"{self.run()}/artifacts/{artifact_id}"

    def commit(self, sha: str) -> str:
        return f"{self.base}/commit/{sha}"

    def blob(self, ref: str, path: str) -> str:
        return f"{self.base}/blob/{ref}/{path}"

    def history(self, ref: str, path: str) -> str:
        return f"{self.base}/commits/{ref}/{path}"


def load_json(path: Optional[Path], warnings: Warnings, what: str) -> Optional[dict]:
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        warnings.add(f"Cannot read {what} '{path}': {error}")
        return None
