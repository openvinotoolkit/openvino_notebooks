def check_jupyter_running():
    """
    Check that Jupyter Lab is running. Raises ValueError if not. This is a utility function 
    for the CI. It is not suitable for use outside of the CI.
    """
    import psutil
    for item in psutil.process_iter():
        if "python" in item.name():
            print(item.cmdline())
            if len([line for line in item.cmdline() if "jupyterlab" in line]) > 0:
                # Jupyter found in list of running processes: return without error
                return
    raise RuntimeError("Jupyter Lab is not running")

def check_install_output():
    """
    Check that the check_install.py script shows "Everything looks good"
    """
    import subprocess
    import sys
    pythonpath = sys.executable
    print(pythonpath)
    result = subprocess.run([pythonpath, "check_install.py"], capture_output=True, universal_newlines=True, shell=True)
    print(result.stdout)
    if not "Everything looks good" in result.stdout:
        raise RuntimeError("check_install failed")
