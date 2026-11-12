"""Schemas and the fragments written by validate_notebooks.py --dashboard_dir."""

import json
from pathlib import Path

from nightly_report.constants import FRAGMENT_FILE_NAME, config_id as make_config_id
from nightly_report.report_common import Warnings, load_json

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
REPORT_SCHEMA_PATH = SCHEMA_DIR / "nightly-report.schema.json"
FRAGMENT_SCHEMA_PATH = SCHEMA_DIR / "fragment.schema.json"


def get_validator(schema_path: Path):
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return None
    return Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))


def schema_errors(validator, data: dict, limit: int = 20) -> list[str]:
    if validator is None:
        return []
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    return [f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}" for error in errors[:limit]]


def load_fragments(fragments_dir: Path, warnings: Warnings) -> list[dict]:
    validator = get_validator(FRAGMENT_SCHEMA_PATH)
    fragments = []
    if not fragments_dir.exists():
        warnings.add(f"Fragments directory '{fragments_dir}' does not exist.")
        return fragments
    for path in sorted(fragments_dir.rglob(FRAGMENT_FILE_NAME)):
        data = load_json(path, warnings, "fragment")
        if not isinstance(data, dict) or not isinstance(data.get("job"), dict):
            continue
        for error in schema_errors(validator, data, limit=5):
            warnings.add(f"Fragment '{path}' does not match the schema: {error}")
        job = data["job"]
        try:
            data["_config_id"] = make_config_id(job["device"], job["os"], job["python"])
        except KeyError:
            warnings.add(f"Fragment '{path}' has no device/os/python; ignored.")
            continue
        data["_dir"] = path.parent
        fragments.append(data)
    return fragments
