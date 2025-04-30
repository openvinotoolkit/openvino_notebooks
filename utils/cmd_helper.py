import logging
import subprocess  # nosec - disable B404:import-subprocess check
import sys
import os
from pathlib import Path
import platform


def clone_repo(repo_url: str, revision: str = None, add_to_sys_path: bool = True) -> Path:
    repo_path = Path(repo_url.split("/")[-1].replace(".git", ""))

    if not repo_path.exists():
        try:
            subprocess.run(["git", "clone", repo_url], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            print(f"Failed to clone the repository: {exc.stderr}")
            raise

        if revision:
            subprocess.Popen(["git", "checkout", revision], cwd=str(repo_path))
    if add_to_sys_path and str(repo_path.resolve()) not in sys.path:
        sys.path.insert(0, str(repo_path.resolve()))

    return repo_path


def optimum_cli(model_id, output_dir, show_command=True, additional_args: dict[str, str] = None, debug_logs=False):
    export_command = f"optimum-cli export openvino --model {model_id} {output_dir}"
    if additional_args is not None:
        for arg, value in additional_args.items():
            export_command += f" --{arg}"
            if value:
                export_command += f" {value}"

    if show_command:
        from IPython.display import Markdown, display

        display(Markdown("**Export command:**"))
        display(Markdown(f"`{export_command}`"))

    transofrmers_loglevel = None
    if debug_logs:
        transofrmers_loglevel = os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        os.environ["TRANSFORMERS_VERBOSITY"] = "debug"

    try:
        subprocess.run(export_command.split(" "), shell=(platform.system() == "Windows"), check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        logger = logging.getLogger()
        logger.exception(exc.stderr)
        if transofrmers_loglevel is not None:
            os.environ["TRANSFORMERS_VERBOSITY"] = transofrmers_loglevel
        raise exc
    finally:
        if transofrmers_loglevel is not None:
            os.environ["TRANSFORMERS_VERBOSITY"] = transofrmers_loglevel
