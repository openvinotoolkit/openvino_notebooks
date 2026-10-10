"""Resolution of notebooks excluded from testing by the skip config yaml or by ignore lists.

Shared by validate_notebooks.py (test jobs) and nightly_report/build_report.py (aggregation job).
"""

from pathlib import Path
from typing import Optional

import yaml

# Same values as nightly_report.constants.SkipReason, kept local so that this module needs only stdlib + yaml
# (it is copied into the Docker image next to validate_notebooks.py without the nightly_report package).
SKIP_CONFIG = "skip_config"
IGNORE_LIST = "ignore_list"


def get_skip_config_notebooks(validation_config: dict, skip_config_file_path: Path) -> list[Path]:
    """Notebooks (repo-root-relative paths as written in the yaml) skipped for the given validation config."""
    ignored_notebooks: list[Path] = []
    if not skip_config_file_path.exists():
        print(f"Skipped notebooks config yaml file does not exist at path '{str(skip_config_file_path)}'.")
        return ignored_notebooks
    with open(skip_config_file_path, "r") as f:
        skipped_notebooks_config = yaml.safe_load(f)
    for skipped_notebook in skipped_notebooks_config:
        skips = skipped_notebook["skips"]
        for skip in skips:
            for key in validation_config.keys():
                if not validation_config[key]:
                    print(f"Warning: validation config argument '{key}' is not provided.")
                if validation_config[key] in skip.get(key, []):
                    ignored_notebooks.append(Path(skipped_notebook["notebook"]))

    return list(set(ignored_notebooks))


def get_ignore_list_notebooks(ignore_list: Optional[list[str]]) -> list[Path]:
    """Notebooks from `--ignore_list` items: `*.txt` files (one repo-root-relative path per line) or notebook paths."""
    ignored_notebooks: list[Path] = []
    if ignore_list is None:
        return ignored_notebooks
    for ignore_item in ignore_list:
        if ignore_item.endswith(".txt"):
            with open(ignore_item, "r") as f:
                ignored_notebooks.extend(Path(line.strip()) for line in f.readlines() if line.strip())
        else:
            ignored_notebooks.append(Path(ignore_item))
    return ignored_notebooks


def resolve_skip_reasons(validation_config: dict, skip_config_file_path: Path, ignore_list: Optional[list[str]]) -> dict[str, str]:
    """Posix repo-root-relative notebook path (as written) -> SKIP_CONFIG / IGNORE_LIST; the skip config takes precedence."""
    skip_config_notebooks = get_skip_config_notebooks(validation_config, skip_config_file_path)
    ignore_list_notebooks = get_ignore_list_notebooks(ignore_list)
    reasons: dict[str, str] = {notebook.as_posix(): IGNORE_LIST for notebook in ignore_list_notebooks}
    reasons.update({notebook.as_posix(): SKIP_CONFIG for notebook in skip_config_notebooks})
    return reasons
