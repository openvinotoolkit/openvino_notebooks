import csv
import json
from pathlib import Path
from typing import Dict
from itertools import product

REPORTS_DIR = "test_reports"


class ValidationMatrix:
    os = ("ubuntu-20.04", "ubuntu-22.04", "windows-2019", "macos-12")
    python = ("3.8", "3.9", "3.10", "3.11")
    device = ("cpu", "gpu")

    @classmethod
    def values(cls):
        return product(cls.device, cls.os, cls.python)


def get_report_file_path(device: str, os: str, python: str) -> Path:
    return Path(REPORTS_DIR) / f"{device}-{os}-{python}" / "test_report.csv"


def get_default_status_dict(notebook_name: str) -> Dict:
    default_status = None

    def _get_python_dict():
        return dict((python, default_status) for python in ValidationMatrix.python)

    def _get_device_dict():
        return dict((device, _get_python_dict()) for device in ValidationMatrix.device)

    return {
        "name": notebook_name,
        "status": dict((os, _get_device_dict()) for os in ValidationMatrix.os),
    }


def write_json_file(filename: str, data: Dict):
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)


def main():
    NOTEBOOKS_STATUS_MAP = {}
    for device, os, python in ValidationMatrix.values():
        if device == "gpu" and not os.startswith("ubuntu"):
            print(f'Tests are not available for "{device}" device and "{os}".')
            continue
        report_file_path = get_report_file_path(device, os, python)
        if not report_file_path.exists():
            print(f'Report file "{report_file_path}" does not exists.')
            continue
        print(f'Processing report file "{report_file_path}".')
        with open(report_file_path, "r") as report_file:
            for row in csv.DictReader(report_file):
                name = row["name"]
                status = row["status"]
                if name not in NOTEBOOKS_STATUS_MAP:
                    NOTEBOOKS_STATUS_MAP[name] = get_default_status_dict(name)
                NOTEBOOKS_STATUS_MAP[name]["status"][os][device][python] = status
    write_json_file(Path(REPORTS_DIR) / "notebooks-status-map.json", NOTEBOOKS_STATUS_MAP)


if __name__ == "__main__":
    main()
