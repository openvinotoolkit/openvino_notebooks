import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product

REPORTS_DIR = "test_reports"


class ValidationMatrix:
    # TODO Uncomment after testing
    # os = ("ubuntu-20.04", "ubuntu-22.04", "windows-2019", "macos-12")
    # python = ("3.8", "3.9", "3.10")
    os = "ubuntu-20.04"
    python = "3.8"

    @classmethod
    def values(cls):
        return product(cls.os, cls.python)


def get_report_file_path(os: str, python: str) -> Path:
    return Path(REPORTS_DIR) / f"{os}-{python}" / "test_report.csv"


def get_default_status_dict(notebook_name: str) -> Dict:
    default_status = None

    def _get_python_status_dict():
        return dict((python, default_status) for python in ValidationMatrix.python)

    return {
        "name": notebook_name,
        "statuses": dict((os, _get_python_status_dict()) for os in ValidationMatrix.os),
    }


def write_json_file(filename: str, data: Dict):
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)


def main():
    NOTEBOOKS_STATUS_MAP = {}
    for os, python in ValidationMatrix.values():
        report_file_path = get_report_file_path(os, python)
        with open(report_file_path, "r") as report_file:
            for row in csv.DictReader(report_file):
                name = row["name"]
                status = row["status"]
                if name not in NOTEBOOKS_STATUS_MAP:
                    NOTEBOOKS_STATUS_MAP[name] = get_default_status_dict(name)
                NOTEBOOKS_STATUS_MAP[name]["statuses"][os][python] = status
    write_json_file("notebooks-status-map.json", NOTEBOOKS_STATUS_MAP)


if __name__ == "__main__":
    main()
