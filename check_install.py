import os
import sys
from pprint import pprint

PYTHON_EXECUTABLE = sys.executable
PYTHON_VERSION = sys.version_info
OS = sys.platform

IN_OPENVINO_ENV = "openvino_env" in sys.executable
SUPPORTED_PYTHON_VERSION = PYTHON_VERSION.major == 3 and (
    PYTHON_VERSION.minor >= 6 and PYTHON_VERSION.minor <= 8
)
GLOBAL_OPENVINO_INSTALLED = "openvino_202" in os.environ.get(
    "LD_LIBRARY_PATH", ""
) + ":".join(sys.path)

try:
    import openvino

    PIP_OPENVINO_INSTALLED = True
except ImportError:
    PIP_OPENVINO_INSTALLED = False

try:
    from openvino.inference_engine import IECore

    OPENVINO_IMPORT = True
except ImportError:
    OPENVINO_IMPORT = False

try:
    import mo_onnx
    import numpy

    NUMPY_VERSION = numpy.__version__
    SUPPORTED_NUMPY_VERSION = NUMPY_VERSION < "1.19"
except ImportError:
    DEVTOOLS_INSTALLED = False
else:
    DEVTOOLS_INSTALLED = True


def show_supported(supported):
    """
    Returns OK (in green) if supported evaluates to True, otherwise NOT OK (in red).
    """
    try:
        from colorama import Fore, Back, Style
        from colorama import init

        init()
        startcolor = Fore.GREEN if supported else Fore.RED
        stopcolor = Style.RESET_ALL
    except:
        startcolor = stopcolor = ""
    output = "OK" if supported else "NOT OK"
    return f"{startcolor}{output}{stopcolor}"


print("System information:")
print(f"Python executable: {PYTHON_EXECUTABLE}")
print(f"OpenVINO environment activated: {show_supported(IN_OPENVINO_ENV)}")
print(
    f"Python version: {PYTHON_VERSION.major}.{PYTHON_VERSION.minor} "
    f"{show_supported(SUPPORTED_PYTHON_VERSION)}"
)
print(
    f"OpenVINO pip package installed: {show_supported(PIP_OPENVINO_INSTALLED)}"
)
print(f"OpenVINO import succeeds: {show_supported(OPENVINO_IMPORT)}")
print(
    f"OpenVINO development tools installed: {show_supported(DEVTOOLS_INSTALLED)}"
)
print(
    f"OpenVINO not installed globally: {show_supported(not GLOBAL_OPENVINO_INSTALLED)}"
)
if DEVTOOLS_INSTALLED:
    print(
        f"Numpy version: {NUMPY_VERSION} {show_supported(SUPPORTED_NUMPY_VERSION)}"
    )
print()

if not PIP_OPENVINO_INSTALLED:
    print(
        f"The OpenVINO PIP package is not installed in this environment. Please\n"
        "follow the README in the same directory as this check_install script or\n"
        "at https://github.com/openvinotoolkit/openvino_notebooks to install OpenVINO."
    )
    sys.exit(0)

if not OPENVINO_IMPORT and OS != "win32" and not GLOBAL_OPENVINO_INSTALLED:
    print(
        f"OpenVINO is installed, but importing fails. This is likely due to a missing\n"
        "libpython.so library for the Python version you are using.\n"
    )
    if OS == "linux":
        print(
            "If you use Python 3.7 on Ubuntu/Debian Linux, you can install the Python\n"
            "libraries with `apt install libpython3.7-dev` (you may need to use `sudo`).\n"
            "On Ubuntu 20, libraries for Python 3.6 and 3.7 are not available to install\n"
            "with apt by default and it is recommended to use Python 3.8.\n"
            "If you have multiple Python version installed, use the full path to the Python\n"
            "executable for creating the virtual environment with a specific Python version.\n"
            "For example: `/usr/bin/python3.8 -m venv openvino_env`. Once you have activated\n"
            "the virtual environment you can type just `python` again.\n"
        )

if not IN_OPENVINO_ENV:
    print(
        "It appears that you are not running Python in an `openvino_env` \n"
        "environment. It is possible use the notebooks in a different \n"
        "environment, but if you run into trouble, please follow the instructions \n"
        "in the README to install and activate the `openvino_env` environment.\n"
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

if not DEVTOOLS_INSTALLED:
    print(
        "OpenVINO development tools are not installed in this Python environment. \n"
        "Please follow the instructions in the README to install `openvino-dev`\n"
    )

elif not SUPPORTED_NUMPY_VERSION:
    print(
        f"You have Numpy version {NUMPY_VERSION}. This may cause issues with model \n"
        "optimization or quantization. Please install `numpy<1.19` with \n"
        "`pip install numpy<1.19`. There may be errors or warnings in the output \n"
        "from pip because of incompatibilities. These should be harmless.\n"
    )
if (
    IN_OPENVINO_ENV
    and PIP_OPENVINO_INSTALLED
    and OPENVINO_IMPORT
    and DEVTOOLS_INSTALLED
    and SUPPORTED_NUMPY_VERSION
    and SUPPORTED_PYTHON_VERSION
    and (not GLOBAL_OPENVINO_INSTALLED)
):
    print("Everything looks good!")
else:
    print(
        "The README.md file is located in the openvino_notebooks directory \n"
        "and at https://github.com/openvinotoolkit/openvino_notebooks"
    )
