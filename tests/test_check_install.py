import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

CHECK_INSTALL = Path(__file__).resolve().parent.parent / "check_install.py"


def load_pip_check():
    # check_install.py runs all its checks at import time, so only load pip_check() from it
    tree = ast.parse(CHECK_INSTALL.read_text())
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pip_check")
    namespace = {"subprocess": subprocess, "sys": sys}
    exec(compile(ast.Module([func], []), str(CHECK_INSTALL), "exec"), namespace)
    return namespace["pip_check"]


def test_pip_check_uses_current_interpreter():
    fake = subprocess.CompletedProcess([], 0, stdout="No broken requirements found.\n")
    with patch("subprocess.run", return_value=fake) as run:
        ok, output = load_pip_check()()
    assert run.call_args[0][0][:4] == [sys.executable, "-m", "pip", "check"]
    assert ok is True
    assert output == ""


def test_pip_check_returns_plain_text_when_requirements_are_broken():
    message = "mypkg 1.0 requires numpy>=2, but you have numpy 1.26.\n"
    fake = subprocess.CompletedProcess([], 1, stdout=message)
    with patch("subprocess.run", return_value=fake):
        ok, output = load_pip_check()()
    assert ok is False
    assert output == message
