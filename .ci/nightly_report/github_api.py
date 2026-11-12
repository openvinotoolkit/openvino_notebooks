"""Minimal GitHub REST API client (stdlib only): run, jobs, artifacts and job logs, with retries; errors become warnings."""

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from nightly_report.report_common import Warnings

JOB_LOG_MAX_BYTES = 64 * 1024 * 1024


class GitHubApi:
    def __init__(self, api_url: str, repository: str, token: Optional[str], warnings: Warnings) -> None:
        self.api_url = api_url.rstrip("/")
        self.repository = repository
        self.token = token
        self.warnings = warnings

    def _request(self, url: str, accept: str = "application/vnd.github+json") -> urllib.request.Request:
        request = urllib.request.Request(url, headers={"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"})
        if self.token:
            # Not sent on redirects: job logs and artifacts redirect to storage URLs that reject foreign credentials.
            request.add_unredirected_header("Authorization", f"Bearer {self.token}")
        return request

    def _fetch(self, url: str, accept: str = "application/vnd.github+json", max_bytes: Optional[int] = None) -> Optional[bytes]:
        request = self._request(url, accept)
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310 - fixed https API URL
                    data = response.read(max_bytes + 1) if max_bytes else response.read()
                    if max_bytes and len(data) > max_bytes:
                        data = data[:max_bytes]
                    return data
            # OSError covers URLError, timeouts and connection resets; HTTPException covers e.g. RemoteDisconnected, IncompleteRead.
            except (OSError, http.client.HTTPException) as error:
                last_error = error
                if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code != 429:
                    break
                if attempt < 2:
                    time.sleep(2**attempt)
        self.warnings.add(f"GitHub API request failed: {url}: {last_error}")
        return None

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        url = f"{self.api_url}/repos/{self.repository}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = self._fetch(url)
        if data is None:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except ValueError as error:
            self.warnings.add(f"GitHub API returned invalid JSON: {url}: {error}")
            return None

    def get_job_log(self, job_id: int) -> Optional[str]:
        """Plain-text log of a completed job (with timestamps), truncated to JOB_LOG_MAX_BYTES."""
        data = self._fetch(f"{self.api_url}/repos/{self.repository}/actions/jobs/{job_id}/logs", accept="*/*", max_bytes=JOB_LOG_MAX_BYTES)
        return data.decode("utf-8", errors="replace") if data is not None else None

    def get_run(self, run_id: int) -> Optional[dict]:
        return self._get(f"actions/runs/{run_id}")

    def _paginate(self, path: str, key: str, params: dict) -> Optional[list]:
        items: list = []
        page = 1
        while True:
            data = self._get(path, {**params, "per_page": 100, "page": page})
            if data is None:
                return items or None
            batch = data.get(key, [])
            items.extend(batch)
            if len(batch) < 100 or len(items) >= data.get("total_count", 0):
                return items
            page += 1

    def get_jobs(self, run_id: int) -> Optional[list]:
        return self._paginate(f"actions/runs/{run_id}/jobs", "jobs", {"filter": "latest"})

    def get_artifacts(self, run_id: int) -> Optional[list]:
        return self._paginate(f"actions/runs/{run_id}/artifacts", "artifacts", {})
