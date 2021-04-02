import sys
from pprint import pprint

PYTHON_EXECUTABLE = sys.executable
PYTHON_VERSION = sys.version_info

IN_OPENVINO_ENV = "openvino_env" in sys.executable
SUPPORTED_PYTHON_VERSION = PYTHON_VERSION.major == 3 and (PYTHON_VERSION.minor == 6 or PYTHON_VERSION.minor == 7)
GLOBAL_OPENVINO_INSTALLED = any(["openvino_202" in path_item for path_item in sys.path])


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
print(f"Python version: {PYTHON_VERSION.major}.{PYTHON_VERSION.minor} {show_supported(SUPPORTED_PYTHON_VERSION)}")
print(f"OpenVINO development tools installed: {show_supported(DEVTOOLS_INSTALLED)}")
if DEVTOOLS_INSTALLED:
    print(f"Numpy version: {NUMPY_VERSION} {show_supported(SUPPORTED_NUMPY_VERSION)}")
print()
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
        "and where you did not add OpenVINO paths to your PATH. \n"
    )
    print("PATH:")
    pprint(sys.path)
    print()
if not DEVTOOLS_INSTALLED:
    print(
        "OpenVINO development tools are not installed in this Python environment. \n"
        "Please follow the instructions in the README to install `openvino-dev`"
    )
elif not SUPPORTED_NUMPY_VERSION:
    print(
        f"You have Numpy version {NUMPY_VERSION}. This may cause issues with model \n"
        "optimization or quantization. Please install `numpy<1.19` with \n"
        "`pip install numpy<1.19`. There may be errors or warnings in the output \n"
        "from pip because of incompatibilities. These should be harmless."
    )
if (
    IN_OPENVINO_ENV
    and DEVTOOLS_INSTALLED
    and SUPPORTED_NUMPY_VERSION
    and SUPPORTED_PYTHON_VERSION
    and (not GLOBAL_OPENVINO_INSTALLED)
):
    print("Everything looks good!")
else:
    print()
    print("The README.md file is located in the openvino_notebooks directory \n"
          "and at https://github.com/openvinotoolkit/openvino_notebooks")
