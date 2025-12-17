import sys
import time
import os
import subprocess  # nosec - disable B404:import-subprocess check
import csv
import json
import shutil
import platform
import psutil
import threading
import queue
import yaml
from clonevirtualenv import clone_virtualenv
import traceback
import tempfile

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, TypedDict
from validation_config import ValidationConfig, validation_config_arg, SkippedNotebook


ROOT = Path(__file__).parents[1]

NOTEBOOKS_DIR = Path("notebooks")

SKIPPED_NOTEBOOKS_CONFIG_FILENAME = "skipped_notebooks.yml"

SEPARATED_VENV_NAME = Path("openvino_venv")


def detect_source_venv_path() -> Path:
    """
    Detect the source virtual environment path based on the current Python executable.

    Returns: Path
    """
    source_venv_path = Path(sys.executable).parent.parent

    print(f"Detecting source virtual environment executable: {sys.executable}", flush=True)
    print(f"Detected source virtual environment path: {source_venv_path}", flush=True)

    return source_venv_path


class NotebookStatus:
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"
    NOT_RUN = "NOT_RUN"
    EMPTY = "EMPTY"


class NotebookReport(TypedDict):
    status: str
    path: Path
    duration: float = 0


TestPlan = dict[Path, NotebookReport]


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--ignore_config", required=False, default=SKIPPED_NOTEBOOKS_CONFIG_FILENAME)
    parser.add_argument("--ignore_list", required=False, nargs="+")
    parser.add_argument("--test_list", required=False, nargs="+")
    parser.add_argument("--os", type=validation_config_arg("os"))
    parser.add_argument("--python", type=validation_config_arg("python"))
    parser.add_argument("--device", type=validation_config_arg("device"))
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--report_dir", default="report")
    parser.add_argument("--keep_artifacts", action="store_true")
    parser.add_argument("--collect_reports", action="store_true")
    parser.add_argument("--move_notebooks_dir")
    parser.add_argument("--job_name")
    parser.add_argument("--upload_to_db")
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout for running single notebook in seconds",
    )
    parser.add_argument(
        "--separate_venv",
        action="store_true",
        help="Use separate virtual environment for each notebook test",
    )
    parser.add_argument(
        "--source_venv_path",
        type=Path,
        help="Path to the source virtual environment to clone for running notebooks",
    )
    parser.add_argument(
        "--cleanup_temp",
        action="store_true",
        help="Cleanup temporary venv directories created during testing before test run is started."
        "Useful when previous test run was interrupted and temporary directories were not removed.",
    )

    return parser.parse_args()


def cleanup_temp_venv_dirs():
    temp_dir = Path(tempfile.gettempdir())
    for item in temp_dir.iterdir():
        if item.is_dir() and item.name.startswith(str(SEPARATED_VENV_NAME)):
            try:
                shutil.rmtree(item)
                print(f"Removed temporary venv directory: {item}", flush=True)
            except Exception as e:
                print(f"Failed to remove temporary venv directory {item}: {e}", flush=True)


def move_notebooks(nb_dir):
    current_notebooks_dir = ROOT / NOTEBOOKS_DIR
    shutil.copytree(current_notebooks_dir, nb_dir)


def collect_python_packages(python_executable: Path, output_file: Path):
    reqs = subprocess.check_output(
        [str(python_executable), "-m", "pip", "freeze"],
        shell=(platform.system() == "Windows"),
    )
    with output_file.open("wb") as f:
        f.write(reqs)


def get_ignored_notebooks_from_yaml(validation_config: ValidationConfig, skip_config_file_path: Path) -> list[Path]:
    ignored_notebooks: list[Path] = []
    if not skip_config_file_path.exists():
        print(f"Skipped notebooks config yaml file does not exist at path '{str(skip_config_file_path)}'.")
        return ignored_notebooks
    with open(skip_config_file_path, "r") as f:
        skipped_notebooks_config: list[SkippedNotebook] = yaml.safe_load(f)
    for skipped_notebook in skipped_notebooks_config:
        skips = skipped_notebook["skips"]
        for skip in skips:
            for key in validation_config.keys():
                if not validation_config[key]:
                    print(f"Warning: validation config argument '{key}' is not provided.")
                if validation_config[key] in skip.get(key, []):
                    ignored_notebooks.append(Path(skipped_notebook["notebook"]))

    return list(set(ignored_notebooks))


def prepare_test_plan(
    validation_config: ValidationConfig, test_list: Optional[list[str]], ignore_config: str, ignore_list: Optional[list[str]], nb_dir: Optional[Path] = None
) -> TestPlan:
    orig_nb_dir = ROOT / NOTEBOOKS_DIR
    notebooks_dir = nb_dir or orig_nb_dir
    notebooks: list[Path] = sorted(list([n for n in notebooks_dir.rglob("**/*.ipynb") if not n.name.startswith("test_")]))

    test_plan: TestPlan = {notebook.relative_to(notebooks_dir): NotebookReport(status="", path=notebook, duration=0) for notebook in notebooks}

    skip_config_file_path = Path(__file__).parents[0] / ignore_config
    ignored_notebooks = get_ignored_notebooks_from_yaml(validation_config, skip_config_file_path)
    if ignore_list is not None:
        for ignore_item in ignore_list:
            if ignore_item.endswith(".txt"):
                # Paths to ignore files are provided to `--ignore_list` argument
                with open(ignore_item, "r") as f:
                    ignored_notebooks.extend(list(map(lambda line: Path(line.strip()), f.readlines())))
            else:
                # Ignored notebooks are provided as several items to `--ignore_list` argument
                ignored_notebooks.append(Path(ignore_item))
    try:
        ignored_notebooks = list(set(map(lambda n: n.relative_to(NOTEBOOKS_DIR), ignored_notebooks)))
    except ValueError:
        raise ValueError(
            f"Ignore list items should be relative to repo root (e.g. 'notebooks/subdir/notebook.ipynb').\nInvalid ignored notebooks: {ignored_notebooks}"
        )
    ignored_notebooks = sorted(ignored_notebooks)
    print(f"Ignored notebooks: {ignored_notebooks}")

    testing_notebooks: list[Path] = []
    if not test_list:
        testing_notebooks = [Path(n) for n in test_plan.keys()]
    elif len(test_list) == 1 and test_list[0].endswith(".txt"):
        with open(test_list[0], "r") as f:
            for line in f.readlines():
                changed_file_path = Path(line.strip())
                if changed_file_path.resolve() == (ROOT / "requirements.txt").resolve():
                    print("requirements.txt changed, check all notebooks")
                    testing_notebooks = [Path(n) for n in test_plan.keys()]
                    break
                if changed_file_path.suffix != ".ipynb":
                    continue
                try:
                    testing_notebook_path = changed_file_path.relative_to(NOTEBOOKS_DIR)
                except ValueError:
                    raise ValueError(
                        "Items in test list file should be relative to repo root (e.g. 'notebooks/subdir/notebook.ipynb').\n"
                        f"Invalid line: {changed_file_path}"
                    )
                testing_notebooks.append(testing_notebook_path)
    elif all(not item.endswith(".txt") for item in test_list):
        # Handle direct notebooks paths passed as arguments
        for notebook_path_str in test_list:
            notebook_path = Path(notebook_path_str.strip())
            if notebook_path.suffix != ".ipynb":
                print(f"Warning: Skipping non-notebook file: {notebook_path}")
                continue
            try:
                testing_notebook_path = notebook_path.relative_to(NOTEBOOKS_DIR)
            except ValueError:
                raise ValueError(
                    "Items in test list should be relative to repo root (e.g. 'notebooks/subdir/notebook.ipynb').\n" f"Invalid notebook path: {notebook_path}"
                )
            testing_notebooks.append(testing_notebook_path)
    else:
        raise ValueError(
            "Testing notebooks should be provided to '--test_list' argument as:\n"
            "  1. A single txt file (e.g., '--test_list notebooks.txt'), OR\n"
            "  2. Multiple notebook paths (e.g., '--test_list notebooks/a.ipynb notebooks/b.ipynb'), OR\n"
            "  3. Empty to test all notebooks.\n"
            f"Received test list: {test_list}"
        )
    testing_notebooks = sorted(list(set(testing_notebooks)))
    print(f"Testing notebooks: {testing_notebooks}")

    for notebook in test_plan:
        if notebook not in testing_notebooks:
            test_plan[notebook]["status"] = NotebookStatus.SKIPPED
        if notebook in ignored_notebooks:
            test_plan[notebook]["status"] = NotebookStatus.SKIPPED
    return test_plan


def clean_test_artifacts(before_test_files: list[Path], after_test_files: list[Path]):
    for file_path in after_test_files:
        if file_path in before_test_files or not file_path.exists():
            continue
        if file_path.is_file():
            try:
                file_path.unlink()
            except Exception:
                pass
        else:
            shutil.rmtree(file_path, ignore_errors=True)


def get_dir_state(dir_path: Path) -> list[Path]:
    """Returns a list containing the directory itself (if exists) and all its contents."""
    if not dir_path.exists():
        return []
    return [dir_path] + sorted(dir_path.rglob("*"))


def get_base_openvino_version() -> str:
    try:
        import openvino as ov

        version = ov.get_version()
        print(f"OpenVINO version in environment before tests started: {version}")
    except ImportError:
        print("OpenVINO is missing in validation environment.")
        version = "Openvino is missing"
    return version


def get_pip_package_version(python_executable: Path, package: str, text_input: str, missing_return: str) -> str:
    command = [str(python_executable), "-m", "pip", "show", package]
    try:
        output = subprocess.check_output(
            command,
            shell=(platform.system() == "Windows"),
            universal_newlines=True,
        )
        version_line = next((line for line in output.splitlines() if line.startswith("Version: ")), None)
        if version_line:
            version = version_line.split("Version: ")[1].strip()
            print(f"{text_input}: {version}")
            return version
        else:
            print(f"{package} is missing in validation environment.")
            return missing_return
    except subprocess.CalledProcessError:
        print(f"{package} is missing in validation environment.")
        return missing_return


def get_dir_size(path: Path) -> int:
    total = 0
    try:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return total


def print_disk_usage(label: str, notebook_dir: Path):
    try:
        # Free disk space
        total, used, free = shutil.disk_usage(notebook_dir.absolute().anchor)

        # Notebook dir size
        nb_dir_size = get_dir_size(notebook_dir)

        # Cache dir size
        cache_dir = Path.home() / ".cache"
        cache_size = get_dir_size(cache_dir)

        print(f"DEBUG [{label}] Free Space: {free} | Notebook Dir: {nb_dir_size} | ~/.cache: {cache_size}", flush=True)
    except Exception as e:
        print(f"Error checking disk usage: {e}")


def clone_venv(source_env_path: Path, target_env_path: Path):
    """
    Clone existing virtual environment to a new location.

    :param source_env_path: source virtual environment path
    :type source_env_path: Path
    :param target_env_path: target virtual environment path
    :type target_env_path: Path
    """

    print(f"Cloning virtual environment from {source_env_path} to " f"{target_env_path}...", flush=True)

    if not source_env_path.exists():
        raise FileNotFoundError(f"Source virtual environment path '{source_env_path}' does not exist.")

    # Validate source environment structure
    if platform.system() == "Windows":
        expected_python = source_env_path / "Scripts" / "python.exe"
        if not expected_python.exists():
            print(f"Warning: Expected python executable not found at {expected_python}", flush=True)
    else:
        expected_python = source_env_path / "bin" / "python"
        if not expected_python.exists():
            print(f"Warning: Expected python executable not found at {expected_python}", flush=True)

    if target_env_path.exists():
        print(
            f"Target virtual environment path '{target_env_path}' already exists. Removing it first...",
            flush=True,
        )
        remove_venv(target_env_path)

    try:
        clone_virtualenv(str(source_env_path), str(target_env_path))
    except Exception as e:
        print(f"Error cloning virtual environment: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise

    print("Virtual environment cloned.", flush=True)

    if platform.system() == "Windows":
        python_exec = target_env_path / "Scripts" / "python.exe"
    else:
        python_exec = target_env_path / "bin" / "python"

    return python_exec.absolute()


def remove_venv(env_path: Path):
    """
    Remove virtual environment at the specified path.

    :param env_path: virtual environment path
    :type env_path: Path
    """
    if env_path.exists() and env_path.is_dir():
        shutil.rmtree(env_path, ignore_errors=True)
        return True
    return False


def read_output_thread(process, output_queue):
    """
    Thread target helper function to read subprocess output in real-time.
    """
    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                output_queue.put(line)
        output_queue.put(None)  # Signal EOF
    except Exception as e:
        print(f"Exception during read_output_thread method: {e}", flush=True)
        output_queue.put(None)  # Signal error/EOF


def kill_process_tree(pid):
    """Kill process tree using platform-specific methods."""
    try:
        if platform.system() == "Windows":
            # On Windows, kill all children in the process group
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            print(f"Killing process tree: parent PID {pid} with {len(children)} children", flush=True)
            for child in children:
                try:
                    print(f"Killing child process PID {child.pid}", flush=True)
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"Could not kill child PID {child.pid}: {e}", flush=True)
            try:
                parent.kill()
                print(f"Killed parent process PID {pid}", flush=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Could not kill parent PID {pid}: {e}", flush=True)
        else:
            # On Unix, kill the entire process group
            import signal

            os.killpg(pid, signal.SIGKILL)
            print(f"Killed process group PID {pid}", flush=True)
    except Exception as e:
        print(f"Error killing process tree PID {pid}: {e}", flush=True)


def run_subprocess_with_timeout(cmd, timeout, shell=False, description="Process"):
    """
    Run a subprocess with real-time output and timeout protection.

    Args:
        cmd: Command to run (list or string)
        timeout: Timeout in seconds
        shell: Whether to use shell=True
        description: Description for logging purposes

    Returns:
        tuple: (return_code, duration)
    """
    # Convert all Path objects to strings in cmd list
    if isinstance(cmd, list):
        cmd = [str(item) for item in cmd]
    print(f"Running {description}: {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    start_time = time.perf_counter()
    process = None
    retcode = None

    # Setup process group creation for proper child process management
    popen_kwargs = {
        "shell": shell,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }

    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        # Use start_new_session instead of preexec_fn to avoid thread-safety warning
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(cmd, **popen_kwargs)

        # Start output reading thread
        output_queue = queue.Queue()
        reader_thread = threading.Thread(target=read_output_thread, args=(process, output_queue), daemon=True)
        reader_thread.start()

        loop_start = time.perf_counter()
        while True:
            # Check timeout FIRST (before any potentially blocking operations)
            if time.perf_counter() - loop_start > timeout:
                print(f"\n{description} timeout reached ({timeout}s), killing process...", flush=True)
                kill_process_tree(process.pid)
                retcode = -42  # Special timeout exit code
                break

            # Check if process finished
            if process.poll() is not None:
                retcode = process.returncode
                break

            # Try to get output with short timeout (non-blocking check)
            try:
                line = output_queue.get(timeout=0.1)
                if line is None:  # EOF signal
                    break
                print(line, end="", flush=True)
            except queue.Empty:
                # No output available, loop continues to check timeout
                continue

        # Drain any remaining output from the queue
        while not output_queue.empty():
            try:
                line = output_queue.get_nowait()
                if line:
                    print(line, end="", flush=True)
            except queue.Empty:
                break

        # Wait for process to finish if not already done
        if retcode is None:
            process.wait()
            retcode = process.returncode

    except Exception as e:
        print(f"\nError running {description}: {e}", flush=True)
        try:
            if process and process.poll() is None:
                kill_process_tree(process.pid)
        except Exception as ex:
            print(f"Error during cleanup: {ex}", flush=True)
        retcode = -1

    duration = time.perf_counter() - start_time
    return retcode, duration


def run_test(
    notebook_path: Path, root, timeout=7200, keep_artifacts=False, report_dir=".", source_venv_path=None
) -> Optional[tuple[str, int, float, str, str]]:
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(notebook_path.parent)
    os.environ["HF_HUB_CACHE"] = str(notebook_path.parent)
    os.environ["TORCH_HOME"] = str(notebook_path.parent)
    os.environ["HF_HOME"] = str(notebook_path.parent)
    os.environ["XDG_CACHE_HOME"] = str(notebook_path.parent / "cache")
    os.environ["PIP_CACHE_DIR"] = str(notebook_path.parent / "pip_cache")
    os.environ["MPLCONFIGDIR"] = str(notebook_path.parent / "mpl_config")
    os.environ["DO_NOT_TRACK"] = "1"
    print(f"RUN {notebook_path.relative_to(root)}", flush=True)
    try:
        relative_path = notebook_path.relative_to(root)
    except ValueError:
        # If notebook_path is not relative to root, use the notebook path as-is
        relative_path = notebook_path
    print(f"RUN {relative_path}", flush=True)
    result = None

    if notebook_path.is_dir():
        print(f'Notebook path "{notebook_path}" is a directory, but path to "*.ipynb" file was expected.')
        return result
    if notebook_path.suffix != ".ipynb":
        print(f'Notebook path "{notebook_path}" should have "*.ipynb" extension.')
        return result

    python_executable = sys.executable

    with tempfile.TemporaryDirectory(prefix=str(SEPARATED_VENV_NAME) + "_") as venv_tmp:
        venv_path = Path(venv_tmp) / SEPARATED_VENV_NAME
        with cd(notebook_path.parent):
            print_disk_usage("BEFORE", Path("."))
            files_before_test = sorted(Path(".").iterdir())
            paddle_before = get_dir_state(Path.home() / ".paddleocr")
            easyocr_before = get_dir_state(Path.home() / ".EasyOCR")
            if source_venv_path:
                try:
                    python_executable = clone_venv(source_venv_path, venv_path)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to create virtual environment for notebook {notebook_path}. Error: {e}")
                    return result

            ov_version_before = get_pip_package_version(python_executable, "openvino", "OpenVINO before notebook execution", "OpenVINO is missing")
            get_pip_package_version(python_executable, "openvino_tokenizers", "OpenVINO Tokenizers before notebook execution", "OpenVINO Tokenizers is missing")
            get_pip_package_version(python_executable, "openvino_genai", "OpenVINO GenAI before notebook execution", "OpenVINO GenAI is missing")
            patched_notebook = Path(f"test_{notebook_path.name}")
            if not patched_notebook.exists():
                print(f'Patched notebook "{patched_notebook}" does not exist.')
                return result

            collect_python_packages(python_executable, report_dir / (patched_notebook.stem + "_env_before.txt"))

            main_command = [python_executable, "-m", "treon", "--verbose", str(patched_notebook)]

            retcode, duration = run_subprocess_with_timeout(
                main_command,
                timeout,
                shell=(platform.system() == "Windows"),
                description=f"Notebook test [{patched_notebook.name}]",
            )

            ov_version_after = get_pip_package_version(python_executable, "openvino", "OpenVINO after notebook execution", "OpenVINO is missing")
            get_pip_package_version(python_executable, "openvino_tokenizers", "OpenVINO Tokenizers after notebook execution", "OpenVINO Tokenizers is missing")
            get_pip_package_version(python_executable, "openvino_genai", "OpenVINO GenAI after notebook execution", "OpenVINO GenAI is missing")
            result = (str(patched_notebook), retcode, duration, ov_version_before, ov_version_after)

            collect_python_packages(python_executable, report_dir / (patched_notebook.stem + "_env_after.txt"))

            if not keep_artifacts:
                clean_test_artifacts(files_before_test, sorted(Path(".").iterdir()))
                clean_test_artifacts(paddle_before, get_dir_state(Path.home() / ".paddleocr"))
                clean_test_artifacts(easyocr_before, get_dir_state(Path.home() / ".EasyOCR"))

            print_disk_usage("AFTER", Path("."))
            print(f"TEST DURATION [{notebook_path.name}]: {duration:.2f} seconds", flush=True)

    return result


def write_csv_report(csv_path, test_report, result_queue):
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "status", "full_path", "duration"])
            writer.writeheader()
            writer.writerows(test_report)
        result_queue.put(("success", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


def finalize_status(failed_notebooks: list[str], timeout_notebooks: list[str], test_plan: TestPlan, report_dir: Path, root: Path) -> int:
    return_status = 0

    if failed_notebooks:
        return_status = 1
        print("FAILED: \n{}".format("\n".join(failed_notebooks)), flush=True)

    if timeout_notebooks:
        print("FAILED BY TIMEOUT: \n{}".format("\n".join(timeout_notebooks)), flush=True)

    test_report = []

    for notebook, status in test_plan.items():
        test_status = status["status"] or NotebookStatus.NOT_RUN
        try:
            full_path_str = str(status["path"].relative_to(root))
        except (ValueError, TypeError):
            full_path_str = str(status["path"].absolute())

        test_report.append({"name": notebook.as_posix(), "status": test_status, "full_path": full_path_str, "duration": status["duration"]})
    print(f"Test report built with {len(test_report)} entries", flush=True)
    csv_path = report_dir / "test_report.csv"
    print(f"Writing test report to: {csv_path.absolute()}", flush=True)
    result_queue = queue.Queue()
    csv_writer_thread = threading.Thread(target=write_csv_report, args=(csv_path, test_report, result_queue), daemon=True)
    csv_writer_thread.start()
    csv_writer_thread.join(timeout=30)

    if csv_writer_thread.is_alive():
        print(f"ERROR: CSV write hung after 30s timeout", flush=True)
        return_status = 1
    else:
        try:
            status, error = result_queue.get_nowait()
            if status == "error":
                print(f"ERROR writing test report: {error}", flush=True)
                return_status = 1
            else:
                print(f"Test report written successfully", flush=True)
        except queue.Empty:
            print(f"ERROR: CSV thread finished but produced no result", flush=True)
            return_status = 1

    return return_status


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)


def write_single_notebook_report(
    base_version: str,
    notebook_name: str,
    status_code: int,
    duration: float,
    ov_version_before: str,
    ov_version_after: str,
    job_name: str,
    device: str,
    saving_dir: Path,
) -> Path:
    report_file = saving_dir / notebook_name.replace(".ipynb", ".json")
    report = {
        "version": base_version,
        "notebook_name": notebook_name.replace("test_", ""),
        "status": status_code,
        "duration": duration,
        "ov_version_before": ov_version_before,
        "ov_version_after": ov_version_after,
        "job_name": job_name,
        "device_used": device,
    }
    with report_file.open("w") as f:
        json.dump(report, f)
    return report_file


def main():
    failed_notebooks = []
    timeout_notebooks = []
    args = parse_arguments()
    reports_dir = Path(args.report_dir).absolute()
    reports_dir.mkdir(exist_ok=True, parents=True)
    notebooks_moving_dir = args.move_notebooks_dir
    root = ROOT

    if args.separate_venv:
        if args.source_venv_path:
            source_venv_path = args.source_venv_path
        else:
            source_venv_path = detect_source_venv_path()
    else:
        source_venv_path = None

    if args.cleanup_temp:
        cleanup_temp_venv_dirs()

    if notebooks_moving_dir is not None:
        notebooks_moving_dir = Path(notebooks_moving_dir).absolute()
        root = notebooks_moving_dir.parent
        move_notebooks(notebooks_moving_dir)
    else:
        notebooks_moving_dir = None

    keep_artifacts = False
    if args.keep_artifacts:
        keep_artifacts = True

    base_version = get_base_openvino_version()

    validation_config = ValidationConfig(os=args.os, python=args.python, device=args.device)

    test_plan = prepare_test_plan(validation_config, args.test_list, args.ignore_config, args.ignore_list, notebooks_moving_dir)

    for notebook, report in test_plan.items():
        if report["status"] == NotebookStatus.SKIPPED:
            continue
        try:
            print("Testing notebook:", str(report["path"]), flush=True)
            test_result = run_test(report["path"], root, args.timeout, keep_artifacts, reports_dir.absolute(), source_venv_path)
        except Exception as e:
            print(f"Error during testing notebook {str(notebook)}: {e}")
            print(traceback.format_exc(), flush=True)
            test_result = [f"test_{report['path'].name}", -1, 0.0, "N/A", "N/A"]
        timing = 0
        if not test_result:
            print(f'Testing notebooks "{str(notebook)}" is not found.')
            report["status"] = NotebookStatus.EMPTY
            report["duration"] = timing
        else:
            patched_notebook, status_code, duration, ov_version_before, ov_version_after = test_result
            if status_code:
                if status_code == -42:
                    status = NotebookStatus.TIMEOUT
                    timeout_notebooks.append(patched_notebook)
                else:
                    status = NotebookStatus.FAILED
                    failed_notebooks.append(patched_notebook)
                report["status"] = status
            else:
                report["status"] = NotebookStatus.SUCCESS if not report["status"] in [NotebookStatus.TIMEOUT, NotebookStatus.FAILED] else report["status"]

            timing += duration
            report["duration"] = timing
            if args.collect_reports:
                job_name = args.job_name or "Unknown"
                device = args.device or "Unknown"
                report_path = write_single_notebook_report(
                    base_version, patched_notebook, status_code, duration, ov_version_before, ov_version_after, job_name, device, reports_dir
                )
                if args.upload_to_db:
                    cmd = [sys.executable, args.upload_to_db, report_path]
                    retcode, duration = run_subprocess_with_timeout(
                        cmd,
                        timeout=15,
                        shell=(platform.system() == "Windows"),
                        description=f"Upload notebook report to DB [{patched_notebook}]",
                    )
                    if retcode != 0:
                        print(f"Database upload failed with exit code {retcode}, duration: {duration:.2f} seconds", flush=True)
                    else:
                        print(f"Database upload succeeded, duration: {duration:.2f} seconds", flush=True)

            if args.early_stop:
                break

    exit_status = finalize_status(failed_notebooks, timeout_notebooks, test_plan, reports_dir, root)
    return exit_status


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
