"""Emit a gh-aw noop before agent startup when the collector selected no work."""

import json
import os
from pathlib import Path


def write_noop_when_idle(status_path: Path, safe_outputs_path: Path) -> bool:
    if not status_path.is_file() or status_path.stat().st_size == 0:
        raise ValueError(f"nightly-remediator status.json is missing or empty: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(status, dict) or not isinstance(status.get("has_work"), bool):
        raise ValueError(f"nightly-remediator status.json has an invalid has_work value: {status_path}")
    if status["has_work"]:
        return False

    reason = status.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"nightly-remediator status.json has no no-work reason: {status_path}")
    safe_outputs_path.parent.mkdir(parents=True, exist_ok=True)
    with safe_outputs_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "noop", "message": f"Nightly remediation skipped: {reason}"}) + "\n")
    return True


def main() -> None:
    digest_dir = os.environ["DIGEST_DIR"]
    safe_outputs = os.environ["GH_AW_SAFE_OUTPUTS"]
    if not digest_dir or not safe_outputs:
        raise ValueError("DIGEST_DIR and GH_AW_SAFE_OUTPUTS must be set")
    write_noop_when_idle(Path(digest_dir) / "status.json", Path(safe_outputs))


if __name__ == "__main__":
    main()
