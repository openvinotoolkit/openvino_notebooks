#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This script launches the OpenVINO notebooks. The notebooks must be
# installed first. To do this, run `python install_notebooks.py` or follow the
# instructions on https://github.com/openvinotoolkit/openvino_notebooks
#
# The script is tested on Windows, macOS and Ubuntu 18.04 and 20.04. It may
# not work on every system though. If there are errors, please
# launch the notebooks manually by activating the virtual environment
# and typing `jupyter lab notebooks` in the directory that contains this
# script.


import subprocess
import sys
from pathlib import Path
import os


pythonpath = sys.executable
curdir = Path(__file__).parent.resolve()
parentdir = curdir.parent

command_line_args = sys.argv[1:]

# If openvino_env is already activated, launch jupyter lab
# This will also start if openvino_env_2 is activated instead of openvino_env
# The assumption is that that is usually intended
if "openvino_env" in pythonpath:
    subprocess.run([pythonpath, "-m", "jupyterlab", "notebooks", *command_line_args])
else:
    if sys.platform == "win32":
        scripts_dir = "Scripts"
    else:
        scripts_dir = "bin"

    # If openvino_env is not activated, search for the openvino_env folder in the
    # current and parent directory and launch the notebooks
    try:
        pythonpath = os.path.normpath(
            os.path.join(curdir, f"openvino_env/{scripts_dir}/python")
        )
        subprocess.run([pythonpath, "-m", "jupyterlab", "notebooks", *command_line_args])
    except:
        try:
            pythonpath = os.path.normpath(
                os.path.join(parentdir, f"openvino_env/{scripts_dir}/python")
            )
            subprocess.run([pythonpath, "-m", "jupyterlab", "notebooks", *command_line_args])
        except SyntaxError:
            subprocess.run([pythonpath, "-m", "jupyterlab", "notebooks"])
        except:
            print(pythonpath)
            print(
                "openvino_env could not be found in the current or parent "
                "directory, or the installation is not complete. Please follow "
                "the instructions on "
                "https://github.com/openvinotoolkit/openvino_notebooks to "
                "install the notebook requirements in a virtual environment.\n\n"
                "After installation, you can also launch the notebooks by "
                "activating the virtual environment manually (see the README "
                "on GitHub, linked above) and typing `jupyter lab notebooks`.\n\n"
                f"Current directory: {curdir}"
                f"Python executable: {sys.executable}"
            )
