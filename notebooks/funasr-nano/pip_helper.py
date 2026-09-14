import sys


def pip_install(*args):
    import subprocess  # nosec - disable B404:import-subprocess check

    cli_args = []
    for arg in args:
        cli_args.extend(str(arg).split(" "))
    subprocess.run([sys.executable, "-m", "pip", "install", *cli_args], check=True)
