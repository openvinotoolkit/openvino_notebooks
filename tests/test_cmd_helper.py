# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for utils/cmd_helper.py — specifically optimum_cli argument handling.

Regression for #3665: paths, model IDs, or additional arg values containing
spaces must be passed to subprocess as single arguments, not split on spaces.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.cmd_helper import optimum_cli


@pytest.fixture
def mock_subprocess():
    with patch("utils.cmd_helper.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock()
        yield mock_run


class TestOptimumCliArguments:
    def test_path_with_spaces_stays_single_argument(self, mock_subprocess):
        """A Windows-style output path with a space must stay one argument."""
        optimum_cli("test-model", "C:\\Users\\Jane Doe\\models", show_command=False)

        cmd = mock_subprocess.call_args[0][0]
        assert cmd == [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            "test-model",
            "C:\\Users\\Jane Doe\\models",
        ]

    def test_onedrive_path_with_spaces_stays_single_argument(self, mock_subprocess):
        """A path with a company OneDrive folder must stay one argument."""
        optimum_cli("test-model", "C:\\Users\\u\\OneDrive - Company\\models", show_command=False)

        cmd = mock_subprocess.call_args[0][0]
        assert cmd[-1] == "C:\\Users\\u\\OneDrive - Company\\models"

    def test_model_id_with_spaces_stays_single_argument(self, mock_subprocess):
        """Model IDs such as 'my org/my model' must stay one argument."""
        optimum_cli("my org/my model", "./out", show_command=False)

        cmd = mock_subprocess.call_args[0][0]
        assert cmd[4] == "my org/my model"
        assert cmd[5] == "./out"

    def test_additional_args_value_with_spaces_stays_single_argument(self, mock_subprocess):
        """Additional arg values with spaces must stay one argument."""
        optimum_cli(
            "test-model",
            "./out",
            additional_args={"task": "text-generation with padding"},
            show_command=False,
        )

        cmd = mock_subprocess.call_args[0][0]
        assert "--task" in cmd
        assert "text-generation with padding" in cmd

    def test_additional_args_empty_value_adds_flag_only(self, mock_subprocess):
        """An additional arg with an empty value should add just the flag."""
        optimum_cli(
            "test-model",
            "./out",
            additional_args={"disable-stateful": ""},
            show_command=False,
        )

        cmd = mock_subprocess.call_args[0][0]
        idx = cmd.index("--disable-stateful")
        assert idx == len(cmd) - 1  # nothing after the flag

    def test_output_dir_path_object_is_stringified(self, mock_subprocess):
        """A pathlib.Path output_dir must be converted to str in the arg list."""
        optimum_cli("test-model", Path("./my out"), show_command=False)

        cmd = mock_subprocess.call_args[0][0]
        assert cmd[-1] == "my out"
        assert isinstance(cmd[-1], str)
