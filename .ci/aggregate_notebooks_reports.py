import csv
import json
from pathlib import Path
from validation_config import ValidationMatrix

REPORTS_DIR = "test_reports"


def find_report_files(device: str, os: str, python: str) -> list[Path]:
    """Find all test_report.csv files for a given device/os/python combination.

    Notebook tests are split into batches (batch_0, batch_1, ...) that run as
    separate CI jobs, each writing its own test_report.csv nested under a
    batch subdirectory (e.g. test_reports/<device>-<os>-<python>/batch_0/test_report.csv)
    so that per-batch artifacts don't collide when merged. Non-batched runs still
    write the report directly under the device-os-python directory, so both
    layouts are searched for here.
    """
    report_dir = Path(REPORTS_DIR) / f"{device}-{os}-{python}"
    if not report_dir.exists():
        return []
    return sorted(report_dir.rglob("test_report.csv"))


def get_default_status_dict(notebook_name: str) -> dict:
    default_status = None

    def _get_python_dict():
        return dict((python, default_status) for python in ValidationMatrix.python)

    def _get_device_dict():
        return dict((device, _get_python_dict()) for device in ValidationMatrix.device)

    return {
        "name": notebook_name,
        "status": dict((os, _get_device_dict()) for os in ValidationMatrix.os),
    }


def write_json_file(filename: str, data: dict):
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)


def main():
    ValidationMatrix.os = tuple(os for os in ValidationMatrix.os if "macos" not in os)

    reports_dir = Path(REPORTS_DIR)
    print(f'Recursive file structure of "{reports_dir}":')
    if reports_dir.exists():
        for path in sorted(reports_dir.rglob("*")):
            print(f"  {path}")
    else:
        print(f'  "{reports_dir}" does not exist.')

    NOTEBOOKS_STATUS_MAP = {}
    processed_reports_count = 0
    for device, os, python in ValidationMatrix.values():
        if device == "gpu" and not os.startswith("ubuntu"):
            print(f'Tests are not available for "{device}" device and "{os}".')
            continue
        report_file_paths = find_report_files(device, os, python)
        if not report_file_paths:
            print(f'No report files found for "{device}-{os}-{python}".')
            continue
        for report_file_path in report_file_paths:
            print(f'Processing report file "{report_file_path}".')
            with open(report_file_path, "r") as report_file:
                for row in csv.DictReader(report_file):
                    name = row["name"]
                    status = row["status"]
                    if name not in NOTEBOOKS_STATUS_MAP:
                        NOTEBOOKS_STATUS_MAP[name] = get_default_status_dict(name)
                    NOTEBOOKS_STATUS_MAP[name]["status"][os][device][python] = status
            processed_reports_count += 1
    print(f"Processed {processed_reports_count} test report file(s).")
    write_json_file(Path(REPORTS_DIR) / "notebooks-status-map.json", NOTEBOOKS_STATUS_MAP)


if __name__ == "__main__":
    main()
