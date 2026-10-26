import sys
from unittest.mock import patch

import pytest

from utils.notebook_utils import pip_install as notebook_pip_install
from utils.pip_helper import pip_install


@pytest.mark.parametrize("installer", [pip_install, notebook_pip_install], ids=["pip_helper", "notebook_utils"])
@pytest.mark.parametrize(
    "args",
    [
        ("requests tqdm",),
        ("requests  tqdm",),
        ("  requests tqdm  ",),
        ("requests\ttqdm",),
        ("requests\ntqdm",),
        ("requests", "tqdm"),
        ("requests", "", "tqdm"),
    ],
)
def test_pip_install_splits_whitespace(installer, args):
    with patch("subprocess.run") as run:
        installer(*args)

    run.assert_called_once()
    assert run.call_args.args[0] == [sys.executable, "-m", "pip", "install", "requests", "tqdm"]
    assert run.call_args.kwargs["check"] is True
