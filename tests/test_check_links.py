import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

CHECK_LINKS = Path(__file__).resolve().parent.parent / ".ci" / "check_links.py"


def run_check_links(tmp_path, responses):
    spec = importlib.util.spec_from_file_location("check_links", CHECK_LINKS)
    check_links = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_links)

    (tmp_path / "README.md").write_text("[link](https://example.com/page)\n")
    check_links.NOTEBOOKS_ROOT = tmp_path
    with patch.object(sys, "argv", ["check_links.py", "README.md"]), patch("time.sleep"), patch("requests.get", side_effect=responses) as get:
        with pytest.raises(SystemExit) as exit_info:
            check_links.main()
    return exit_info.value.code, get.call_count


def response(status_code):
    return MagicMock(status_code=status_code)


def test_retries_after_transient_connection_error(tmp_path):
    code, calls = run_check_links(tmp_path, [requests.exceptions.ConnectionError("reset by peer"), response(200)])
    assert code == 0
    assert calls == 2


def test_retries_after_timeout(tmp_path):
    code, calls = run_check_links(tmp_path, [requests.exceptions.ReadTimeout("timed out"), response(200)])
    assert code == 0
    assert calls == 2


def test_retries_after_server_error(tmp_path):
    code, calls = run_check_links(tmp_path, [response(503), response(200)])
    assert code == 0
    assert calls == 2


def test_fails_when_connection_keeps_failing(tmp_path):
    errors = [requests.exceptions.ConnectionError("down")] * 3
    code, calls = run_check_links(tmp_path, errors)
    assert code == 1
    assert calls == 3


def test_does_not_retry_not_found(tmp_path):
    code, calls = run_check_links(tmp_path, [response(404), response(200)])
    assert code == 1
    assert calls == 1
