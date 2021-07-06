#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This script installs the OpenVINO notebooks in an openvino_env
# virtual environment. If an openvino_env directory exists in the
# current directory, the notebooks will be installed there. If not,
# a new virtualenv will be created there.
#
# The script is tested on Windows, macOS and Ubuntu 18.04 and 20.04. It may
# not work on every system though. If there are errors, please follow the
# instructions on https://github.com/openvinotoolkit/openvino_notebooks to
# install the notebooks manually. If you find a bug or issue, please report
# it at https://github.com/openvinotoolkit/openvino_notebooks/issues
# Suggestions are welcome at
# https://github.com/openvinotoolkit/openvino_notebooks/discussions


import subprocess
import sys
from pathlib import Path
import os

pythonpath = sys.executable
curdir = Path(__file__).parent.resolve()
parentdir = curdir.parent

scripts_dir = "Scripts" if sys.platform == "win32" else "bin"

# -----------------------------------------------------------------------#
# Check for supported python version.                                    #
# -----------------------------------------------------------------------#

PYTHON_VERSION = sys.version_info
SUPPORTED_PYTHON_VERSION = PYTHON_VERSION.major == 3 and (
    PYTHON_VERSION.minor >= 6 and PYTHON_VERSION.minor <= 8
)

if not SUPPORTED_PYTHON_VERSION:
    error_message = f"Python {PYTHON_VERSION.major}.{PYTHON_VERSION.minor} is not currently " \
                    "supported by OpenVINO.\n" \
                    "Please install python 3.6, 3.7, or 3.8."

    if sys.platform == "win32":
        error_message += "\nIf you used the python.org installer to install multiple versions\n" \
                         "of Python, you can choose a specific version with `py`\n" \
                         "For example, type: `py -3.7 install_notebooks.py`"
    else:
        error_message += "\nIf you installed multiple versions of python, call this script \n" \
                         "with the path to the Python executable. For example: \n" \
                         "`/usr/bin/python3.7` install_notebooks.py"
    print(error_message)
    sys.exit()


# -----------------------------------------------------------------------#
# Check for confirmation to install notebooks                            #
# -----------------------------------------------------------------------#


if "-y" not in sys.argv:
    agree = input("OpenVINO notebook requirements will be installed in "
                  "an 'openvino_env' environment in the current directory.\n"
                  "The environment will be created if it does not exist. Do "
                  "you want to continue? [Y/N]\n")
    if agree.upper() != "Y":
        sys.exit()

# -----------------------------------------------------------------------#
# If openvino_env exists in the current directory, try to install there. #
# If openvino_env does not exist, create it.                             #
# -----------------------------------------------------------------------#

if "openvino_env" not in pythonpath:
    dirlist = [item.name for item in Path(".").iterdir()]

    if "openvino_env" not in dirlist:
        print("Creating openvino_env virtualenv...")
        subprocess.run([pythonpath, "-m", "venv", "openvino_env"])
        print("openvino_env created.")
    pythonpath = os.path.normpath(os.path.join(curdir, f"openvino_env/{scripts_dir}/python"))

try:
    print("Upgrading pip...")
    subprocess.run([pythonpath, "-m", "pip", "install", "--upgrade", "pip"], shell=False)
    print("Installing all requirements. This may take a while...")
    subprocess.run([pythonpath, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt",
                    "--use-deprecated", "legacy-resolver"], shell=False)
    subprocess.run([pythonpath, "-m", "ipykernel", "install", "--user", "--name", "openvino_env"])
except Exception as e:
    print("\nInstallation failed. Please follow the instructions on "
          "https://github.com/openvinotoolkit/openvino_notebooks\n"
          "to install the notebooks manually. The error message is:")
    print(e)
else:
    print("\nInstallation succeeded! You can launch the notebooks with "
          "`python launch_notebooks.py`")
