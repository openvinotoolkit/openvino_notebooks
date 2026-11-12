import sys
from unittest.mock import patch

import pytest

from utils.notebook_utils import pip_install


@pytest.mark.parametrize("system", ["Windows", "Linux", "Darwin"])
def test_pip_install_does_not_use_shell(system):
    # with shell=True cmd.exe treats ">" and "<" in version specifiers as redirections
    with patch("platform.system", return_value=system), patch("subprocess.run") as run:
        pip_install("-q", "openvino>=2026.0.0", "setuptools<82")
    assert run.call_args[0][0] == [sys.executable, "-m", "pip", "install", "-q", "openvino>=2026.0.0", "setuptools<82"]
    assert not run.call_args.kwargs.get("shell", False)


def test_pip_install_splits_arguments_given_as_one_string():
    with patch("subprocess.run") as run:
        pip_install("-q openvino-genai", "nncf>=2.17.0")
    assert run.call_args[0][0] == [sys.executable, "-m", "pip", "install", "-q", "openvino-genai", "nncf>=2.17.0"]
