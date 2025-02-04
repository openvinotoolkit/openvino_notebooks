import sys
import time
import os
import subprocess  # nosec - disable B404:import-subprocess check
import csv
import json
import shutil
import platform
import yaml

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict
from validation_config import ValidationConfig, validation_config_arg, SkippedNotebook


ROOT = Path(__file__).parents[1]

NOTEBOOKS_DIR = Path("notebooks")

SKIPPED_NOTEBOOKS_CONFIG_FILENAME = "skipped_notebooks.yml"


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


TestPlan = Dict[Path, NotebookReport]


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
    return parser.parse_args()


def move_notebooks(nb_dir):
    current_notebooks_dir = ROOT / NOTEBOOKS_DIR
    shutil.copytree(current_notebooks_dir, nb_dir)


def collect_python_packages(output_file: Path):
    reqs = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        shell=(platform.system() == "Windows"),
    )
    with output_file.open("wb") as f:
        f.write(reqs)


def get_ignored_notebooks_from_yaml(validation_config: ValidationConfig, skip_config_file_path: Path) -> List[Path]:
    ignored_notebooks: List[Path] = []
    if not skip_config_file_path.exists():
        print(f"Skipped notebooks config yaml file does not exist at path '{str(skip_config_file_path)}'.")
        return ignored_notebooks
    with open(skip_config_file_path, "r") as f:
        skipped_notebooks_config: List[SkippedNotebook] = yaml.safe_load(f)
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
    validation_config: ValidationConfig, test_list: Optional[List[str]], ignore_config: str, ignore_list: Optional[List[str]], nb_dir: Optional[Path] = None
) -> TestPlan:
    orig_nb_dir = ROOT / NOTEBOOKS_DIR
    notebooks_dir = nb_dir or orig_nb_dir
    notebooks: List[Path] = sorted(list([n for n in notebooks_dir.rglob("**/*.ipynb") if not n.name.startswith("test_")]))

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

    testing_notebooks: List[Path] = []
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
    else:
        raise ValueError(
            "Testing notebooks should be provided to '--test_list' argument as a txt file or should be empty to test all notebooks.\n"
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


def clean_test_artifacts(before_test_files: List[Path], after_test_files: List[Path]):
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


def get_base_openvino_version() -> str:
    try:
        import openvino as ov

        version = ov.get_version()
        print(f"OpenVINO version in environment before tests started: {version}")
    except ImportError:
        print("OpenVINO is missing in validation environment.")
        version = "Openvino is missing"
    return version


def get_pip_package_version(package, text_input: str, missing_return: str) -> str:
    try:
        from importlib import metadata

        version = metadata.version(package)
        print(f"{text_input}: {version}")
    except metadata.PackageNotFoundError:
        print(f"{package} is missing in validation environment.")
        version = missing_return
    return version


def run_test(notebook_path: Path, root, timeout=7200, keep_artifacts=False, report_dir=".") -> Optional[Tuple[str, int, float, str, str]]:
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(notebook_path.parent)
    os.environ["HF_HUB_CACHE"] = str(notebook_path.parent)
    os.environ["TORCH_HOME"] = str(notebook_path.parent)
    os.environ["DO_NOT_TRACK"] = "1"
    print(f"RUN {notebook_path.relative_to(root)}", flush=True)
    result = None

    if notebook_path.is_dir():
        print(f'Notebook path "{notebook_path}" is a directory, but path to "*.ipynb" file was expected.')
        return result
    if notebook_path.suffix != ".ipynb":
        print(f'Notebook path "{notebook_path}" should have "*.ipynb" extension.')
        return result

    with cd(notebook_path.parent):
        files_before_test = sorted(Path(".").iterdir())
        ov_version_before = get_pip_package_version("openvino", "OpenVINO before notebook execution", "OpenVINO is missing")
        get_pip_package_version("openvino_tokenizers", "OpenVINO Tokenizers before notebook execution", "OpenVINO Tokenizers is missing")
        get_pip_package_version("openvino_genai", "OpenVINO GenAI before notebook execution", "OpenVINO GenAI is missing")
        patched_notebook = Path(f"test_{notebook_path.name}")
        if not patched_notebook.exists():
            print(f'Patched notebook "{patched_notebook}" does not exist.')
            return result

        collect_python_packages(report_dir / (patched_notebook.stem + "_env_before.txt"))

        main_command = [sys.executable, "-m", "treon", "--verbose", str(patched_notebook)]
        start = time.perf_counter()
        try:
            retcode = subprocess.run(
                main_command,
                shell=(platform.system() == "Windows"),
                timeout=timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            retcode = -42
        duration = time.perf_counter() - start
        ov_version_after = get_pip_package_version("openvino", "OpenVINO after notebook execution", "OpenVINO is missing")
        get_pip_package_version("openvino_tokenizers", "OpenVINO Tokenizers after notebook execution", "OpenVINO Tokenizers is missing")
        get_pip_package_version("openvino_genai", "OpenVINO GenAI after notebook execution", "OpenVINO GenAI is missing")
        result = (str(patched_notebook), retcode, duration, ov_version_before, ov_version_after)

        if not keep_artifacts:
            clean_test_artifacts(files_before_test, sorted(Path(".").iterdir()))
        collect_python_packages(report_dir / (patched_notebook.stem + "_env_after.txt"))

    return result


def finalize_status(failed_notebooks: List[str], timeout_notebooks: List[str], test_plan: TestPlan, report_dir: Path, root: Path) -> int:
    return_status = 0
    if failed_notebooks:
        return_status = 1
        print("FAILED: \n{}".format("\n".join(failed_notebooks)))
    if timeout_notebooks:
        print("FAILED BY TIMEOUT: \n{}".format("\n".join(timeout_notebooks)))
    test_report = []
    for notebook, status in test_plan.items():
        test_status = status["status"] or NotebookStatus.NOT_RUN
        test_report.append(
            {"name": notebook.as_posix(), "status": test_status, "full_path": str(status["path"].relative_to(root)), "duration": status["duration"]}
        )
    with (report_dir / "test_report.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "status", "full_path", "duration"])
        writer.writeheader()
        writer.writerows(test_report)
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
    reports_dir = Path(args.report_dir)
    reports_dir.mkdir(exist_ok=True, parents=True)
    notebooks_moving_dir = args.move_notebooks_dir
    root = ROOT
    if notebooks_moving_dir is not None:
        notebooks_moving_dir = Path(notebooks_moving_dir)
        root = notebooks_moving_dir.parent
        move_notebooks(notebooks_moving_dir)

    keep_artifacts = False
    if args.keep_artifacts:
        keep_artifacts = True

    base_version = get_base_openvino_version()

    validation_config = ValidationConfig(os=args.os, python=args.python, device=args.device)

    test_plan = prepare_test_plan(validation_config, args.test_list, args.ignore_config, args.ignore_list, notebooks_moving_dir)

    for notebook, report in test_plan.items():
        if report["status"] == NotebookStatus.SKIPPED:
            continue
        test_result = run_test(report["path"], root, args.timeout, keep_artifacts, reports_dir.absolute())
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
                    print(f"\nUploading {report_path} to database. CMD: {cmd}")
                    try:
                        dbprocess = subprocess.Popen(
                            cmd, shell=(platform.system() == "Windows"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
                        )
                        for line in dbprocess.stdout:
                            sys.stdout.write(line)
                            sys.stdout.flush()
                    except subprocess.CalledProcessError as e:
                        print(e.output)

            if args.early_stop:
                break

    exit_status = finalize_status(failed_notebooks, timeout_notebooks, test_plan, reports_dir, root)
    return exit_status


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
