import csv
import json

import aggregate_notebooks_reports
from validation_config import ValidationMatrix


def test_aggregate_python_314_batch_reports(tmp_path, monkeypatch):
    reports = tmp_path / "test_reports"
    monkeypatch.setattr(aggregate_notebooks_reports, "REPORTS_DIR", str(reports))
    monkeypatch.setattr(ValidationMatrix, "os", ValidationMatrix.os)
    for os_name, version, batch, status in (
        ("ubuntu-22.04", "3.14", "batch_0", "NOT_RUN"),
        ("ubuntu-22.04", "3.14", "batch_1", "SUCCESS"),
        ("ubuntu-22.04", "3.11", "batch_0", "FAILED"),
        ("windows-2022", "3.14", "batch_0", "TIMEOUT"),
    ):
        report = reports / f"cpu-{os_name}-{version}" / batch / "test_report.csv"
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["name", "status"])
            writer.writeheader()
            writer.writerow({"name": "a/a.ipynb", "status": status})
    aggregate_notebooks_reports.main()
    status = json.loads((reports / "notebooks-status-map.json").read_text(encoding="utf-8"))["a/a.ipynb"]["status"]
    assert status["ubuntu-22.04"]["cpu"]["3.14"] == "SUCCESS"
    assert status["ubuntu-22.04"]["cpu"]["3.11"] == "FAILED"
    assert status["windows-2022"]["cpu"]["3.14"] == "TIMEOUT"
