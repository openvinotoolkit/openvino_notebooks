import os
import subprocess  # nosec - disable B404:import-subprocess check
import sys
from pathlib import Path
from pprint import pprint

import pip

try:
    from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel
except:
    print(
        "Importing Jupyter failed. Please follow the installation instructions "
        "in the README in the same directory as this script or "
        "at https://github.com/openvinotoolkit/openvino_notebooks."
    )
    sys.exit()


def show_supported(supported):
    """
    Returns OK (in green) if supported evaluates to True, otherwise NOT OK (in red).
    """
    try:
        from colorama import Fore, Style, init

        init()
        startcolor = Fore.GREEN if supported else Fore.RED
        stopcolor = Style.RESET_ALL
    except:
        startcolor = stopcolor = ""
    output = "OK" if supported else "NOT OK"
    return f"{startcolor}{output}{stopcolor}"


def pip_check():
    result = subprocess.run(["pip", "check"], universal_newlines=True, stdout=subprocess.PIPE)
    if "No broken requirements found" in result.stdout:
        return True, ""
    else:
        return False, result


def kernel_check():
    try:
        kernel = KernelSpecManager().get_kernel_spec("openvino_env")
    except NoSuchKernel:
        return False, ""
    kernel_python = kernel.argv[0]
    return True, kernel_python


PYTHON_EXECUTABLE = sys.executable
PYTHON_VERSION = sys.version_info
PIP_VERSION = pip.__version__
OS = sys.platform
KERNEL_INSTALLED, KERNEL_PYTHON = kernel_check()
NO_BROKEN_REQUIREMENTS, PIP_CHECK_OUTPUT = pip_check()

CORRECT_KERNEL_PYTHON = PYTHON_EXECUTABLE == KERNEL_PYTHON

IN_OPENVINO_ENV = "openvino_env" in sys.executable
SUPPORTED_PYTHON_VERSION = PYTHON_VERSION.major == 3 and (PYTHON_VERSION.minor >= 8 and PYTHON_VERSION.minor <= 11)
GLOBAL_OPENVINO_INSTALLED = "openvino_202" in os.environ.get("LD_LIBRARY_PATH", "") + ":".join(sys.path)


try:
    import openvino

    PIP_OPENVINO_INSTALLED = True
except ImportError:
    PIP_OPENVINO_INSTALLED = False

try:
    import openvino
    from openvino.runtime import Core

    OPENVINO_IE_VERSION = openvino.runtime.get_version()
    OPENVINO_SOURCE_ROOT = str(Path(openvino.__file__).parent)
    OPENVINO_IMPORT = True
except ImportError:
    OPENVINO_IMPORT = False


print("System information:")
print(f"Python executable: {PYTHON_EXECUTABLE}")
print(f"Pip version: {PIP_VERSION}")
if OPENVINO_IMPORT:
    print(f"OpenVINO source: {OPENVINO_SOURCE_ROOT}")
    print(f"OpenVINO IE version: {OPENVINO_IE_VERSION}")
print(f"OpenVINO environment activated: {show_supported(IN_OPENVINO_ENV)}")
print(f"Jupyter kernel installed for openvino_env: {show_supported(KERNEL_INSTALLED)}")
if KERNEL_INSTALLED:
    print(f"Jupyter kernel Python executable: {KERNEL_PYTHON}")
    print("Jupyter kernel Python and OpenVINO environment Python match: " f"{show_supported(CORRECT_KERNEL_PYTHON)}")
print(f"Python version: {PYTHON_VERSION.major}.{PYTHON_VERSION.minor} " f"{show_supported(SUPPORTED_PYTHON_VERSION)}")
print(f"OpenVINO pip package installed: {show_supported(PIP_OPENVINO_INSTALLED)}")
print(f"OpenVINO import succeeds: {show_supported(OPENVINO_IMPORT)}")
print(f"OpenVINO not installed globally: {show_supported(not GLOBAL_OPENVINO_INSTALLED)}")

print(f"No broken requirements: {show_supported(NO_BROKEN_REQUIREMENTS)}")
print()

if not PIP_OPENVINO_INSTALLED:
    print(
        "The OpenVINO PIP package is not installed in this environment. Please\n"
        "follow the README in the same directory as this check_install script or\n"
        "at https://github.com/openvinotoolkit/openvino_notebooks to install OpenVINO."
    )
    sys.exit(0)

if not OPENVINO_IMPORT and OS != "win32" and not GLOBAL_OPENVINO_INSTALLED:
    print("OpenVINO is installed, but importing fails. This is likely due to a missing\n" "libpython.so library for the Python version you are using.\n")
    if OS == "linux":
        print(
            "If you have multiple Python version installed, use the full path to the Python\n"
            "executable for creating the virtual environment with a specific Python version.\n"
            "For example: `/usr/bin/python3.10 -m venv openvino_env`. Once you have activated\n"
            "the virtual environment you can type just `python` again.\n"
        )

if not IN_OPENVINO_ENV:
    print(
        "It appears that you are not running Python in an `openvino_env` \n"
        "environment. It is possible use the notebooks in a different \n"
        "environment, but if you run into trouble, please follow the instructions \n"
        "in the README to install and activate the `openvino_env` environment.\n"
    )

if not CORRECT_KERNEL_PYTHON:
    print(
        "The Python version in openvino_env does not match the openvino_env Jupyter kernel.\n"
        "This may not be an issue. If you experience issues, please follow the instructions\n"
        "in the README to reinstall the kernel."
    )
if GLOBAL_OPENVINO_INSTALLED:
    print(
        "It appears that you installed OpenVINO globally (for example with \n"
        "the OpenVINO installer, or a package manager). \n"
        "This may cause conflicts with the OpenVINO environment installed by \n"
        "pip install. If you encounter issues, please make sure to start the \n"
        "notebooks from a terminal where you did not run setupvars.sh/setupvars.bat, \n"
        "and where you did not add OpenVINO paths to your PATH or LD_LIBRARY_PATH. \n"
    )
    if OS == "win32":
        print("PATH:")
        pprint(sys.path)
        print()
    else:
        print("LD_LIBRARY_PATH:")
        pprint(os.environ.get("LD_LIBRARY_PATH", ""))
        print()
        print(
            "You may have added the command to source setuptools.sh to your \n"
            ".bashrc, or added the OpenVINO paths to LD_LIBRARY_PATH there.\n"
            "You can delete the lines from .bashrc and open a new terminal window\n"
            "or temporarily reset your LD_LIBRARY_PATH by executing\n"
            "`export LD_LIBRARY_PATH=` in your current terminal.\n"
        )

if (not OPENVINO_IMPORT) and (OS == "win32" and PIP_OPENVINO_INSTALLED):
    print()
    print("Importing OpenVINO failed. ")
    if os.environ.get("CONDA_PREFIX") is not None:
        print(
            "To use openvino in a conda environment, you may need to "
            "adjust your PATH. See step 6 in \n"
            "https://github.com/openvinotoolkit/openvino_notebooks/wiki/Conda "
        )
    else:
        print(
            "Importing OpenVINO failed. If you installed Python from the \n"
            "Windows Store, please try with the Python installer from python.org.\n"
            "See https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows"
        )

if not NO_BROKEN_REQUIREMENTS:
    print()
    print("`pip check` shows broken requirements:")
    print(PIP_CHECK_OUTPUT)

print()
if (
    IN_OPENVINO_ENV
    and PIP_OPENVINO_INSTALLED
    and OPENVINO_IMPORT
    and SUPPORTED_PYTHON_VERSION
    and KERNEL_INSTALLED
    and CORRECT_KERNEL_PYTHON
    and (not GLOBAL_OPENVINO_INSTALLED)
):
    if NO_BROKEN_REQUIREMENTS:
        print("Everything looks good!")
    else:
        print("Summary: The installation looks good, but there are conflicting requirements.")
else:
    print("The README.md file is located in the openvino_notebooks directory \n" "and at https://github.com/openvinotoolkit/openvino_notebooks")
if not NO_BROKEN_REQUIREMENTS:
    print("Broken requirements are often harmless, but could cause issues.")
