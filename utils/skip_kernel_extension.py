# https://github.com/RobbeSneyders/Jupyter-skip-extension/blob/e77e3708ebb37a3ff0e851fd773adc851d7c1d6a/skip_kernel_extension.py

def skip(line, cell=None):
    """Skips execution of the current line/cell."""
    if eval(line):
        return

    get_ipython().ex(cell)


def load_ipython_extension(shell):
    """Registers the skip magic when the extension loads."""
    shell.register_magic_function(skip, "line_cell")


def unload_ipython_extension(shell):
    """Unregisters the skip magic when the extension unloads."""
    del shell.magics_manager.magics["cell"]["skip"]
