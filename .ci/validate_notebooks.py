import sys
import time
import os
import subprocess  # nosec - disable B404:import-subprocess check
import csv
import json
import shutil
import platform
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple, TypedDict


ROOT = Path(__file__).parents[1]


class NotebookReport(TypedDict):
    status: str
    path: Path
    duration: float = 0


TestPlan = Dict[Path, NotebookReport]


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--ignore_list", required=False, nargs="+")
    parser.add_argument("--test_list", required=False, nargs="+")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--report_dir", default="report")
    parser.add_argument("--keep_artifacts", action="store_true")
    parser.add_argument("--collect_reports", action="store_true")
    parser.add_argument("--move_notebooks_dir")
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout for running single notebook in seconds",
    )
    return parser.parse_args()


def find_notebook_dir(path, root):
    for parent in path.parents:
        if root == parent.parent:
            return parent.relative_to(root)
    return None


def move_notebooks(nb_dir):
    current_notebooks_dir = ROOT / "notebooks"
    shutil.copytree(current_notebooks_dir, nb_dir)


def get_notebooks_subdir(changed_path, orig_nb_dir) -> Optional[Path]:
    if (orig_nb_dir / changed_path).exists() and (orig_nb_dir / changed_path).is_dir():
        notebook_subdir = orig_nb_dir / changed_path
        if not list(notebook_subdir.rglob("**/*.ipynb")):
            notebook_subdir = None
        else:
            notebook_subdir = notebook_subdir.relative_to(orig_nb_dir)
        print(notebook_subdir)
    else:
        notebook_subdir = find_notebook_dir(changed_path.resolve(), orig_nb_dir.resolve())
    return notebook_subdir


def prepare_test_plan(test_list: List[str], ignore_list: List[str], nb_dir: Optional[Path] = None) -> TestPlan:
    orig_nb_dir = ROOT / "notebooks"
    notebooks_dir = orig_nb_dir if nb_dir is None else nb_dir
    notebooks = sorted(list([n for n in notebooks_dir.rglob("**/*.ipynb") if not n.name.startswith("test_")]))

    statuses = {notebook.relative_to(notebooks_dir): NotebookReport(status="", path=notebook, duration=0) for notebook in notebooks}

    test_list = test_list or statuses.keys()
    ignored_notebooks = []
    if ignore_list is not None:
        for ignore_item in ignore_list:
            if ignore_item.endswith(".txt"):
                # Paths to ignore files are provided to `--ignore_list` argument
                with open(ignore_item, "r") as f:
                    ignored_notebooks.extend(list(map(lambda x: x.strip(), f.readlines())))
            else:
                # Notebooks list is provided to `--ignore_list` argument
                ignored_notebooks.append(ignore_item)
        print(f"ignored notebooks: {ignored_notebooks}")

    testing_notebooks = []
    if len(test_list) == 1 and test_list[0].endswith(".txt"):
        with open(test_list[0], "r") as f:
            for line in f.readlines():
                changed_file_path = Path(line.strip())
                if changed_file_path.resolve() == (ROOT / "requirements.txt").resolve():
                    print("requirements.txt changed, check all notebooks")
                    testing_notebooks = statuses.keys()
                    break
                if changed_file_path.suffix == ".md":
                    continue
                # notebook_subdir = get_notebooks_subdir(changed_file_path, orig_nb_dir)
                # if notebook_subdir is not None:
                # testing_notebooks.append(notebook_subdir)
                testing_notebooks.append(changed_file_path.relative_to(orig_nb_dir))
    else:
        for test_item in test_list:
            # notebook_subdir = get_notebooks_subdir(Path(test_item), orig_nb_dir)
            # if notebook_subdir is not None:
            #     testing_notebooks.append(notebook_subdir)
            try:
                testing_notebook_path = Path(test_item).relative_to(orig_nb_dir)
            except ValueError:
                testing_notebook_path = Path(test_item)
            testing_notebooks.append(testing_notebook_path)
    test_list = set(testing_notebooks)
    print(f"test notebooks: {test_list}")

    ignore_list = set(map(lambda x: Path(x), ignored_notebooks))
    for notebook in statuses:
        if notebook not in test_list:
            statuses[notebook]["status"] = "SKIPPED"
        if notebook in ignore_list:
            statuses[notebook]["status"] = "SKIPPED"
    return statuses


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


def run_test(notebook_path: Path, root, timeout=7200, keep_artifacts=False, report_dir=".") -> Optional[Tuple[str, int, float]]:
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(notebook_path.parent)
    print(f"RUN {notebook_path.relative_to(root)}", flush=True)
    result = None

    if notebook_path.is_dir():
        print(f'Notebook path "{notebook_path}" is a directory, but path to "*.ipynb" file was expected.')
        return result
    if notebook_path.suffix != ".ipynb":
        print(f'Notebook path "{notebook_path}" should have "*.ipynb" extension.')
        return result

    with cd(notebook_path.parent):
        patched_notebook = Path(f"test_{notebook_path.name}")
        if not patched_notebook.exists():
            print(f'Patched notebook "{patched_notebook}" does not exist.')
            return result

        print("Packages before notebook run")
        reqs = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            shell=(platform.system() == "Windows"),
        )
        reqs_before_file = report_dir / (patched_notebook.stem + "_env_before.txt")
        with reqs_before_file.open("wb") as f:
            f.write(reqs)
        with reqs_before_file.open("r") as f:
            print(f.read())

        main_command = [sys.executable, "-m", "treon", str(patched_notebook)]
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
        result = (str(patched_notebook), retcode, duration)

        if not keep_artifacts:
            clean_test_artifacts([patched_notebook], sorted(Path(".").iterdir()))
        print("Packages after notebook run")
        reqs = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            shell=(platform.system() == "Windows"),
        )
        reqs_after_file = report_dir / (patched_notebook.stem + "_env_after.txt")
        with reqs_after_file.open("wb") as f:
            f.write(reqs)
        with reqs_after_file.open("r") as f:
            print(f.read())
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
        test_status = status["status"] or "NOT_RUN"
        test_report.append({"name": notebook, "status": test_status, "full_path": str(status["path"].relative_to(root)), "duration": status["duration"]})
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


def write_single_notebook_report(notebook_name, status, duration, saving_dir):
    report_file = saving_dir / notebook_name.replace(".ipynb", ".json")
    report = {
        "notebook_name": notebook_name.replace("test_", ""),
        "status": status,
        "duration": duration,
    }
    with report_file.open("w") as f:
        json.dump(report, f)


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

    test_plan = prepare_test_plan(args.test_list, args.ignore_list, notebooks_moving_dir)
    for notebook, report in test_plan.items():
        if report["status"] == "SKIPPED":
            continue
        test_result = run_test(report["path"], root, args.timeout, keep_artifacts, reports_dir.absolute())
        timing = 0
        if not test_result:
            print(f'Testing notebooks "{str(notebook)}" is not found.')
            report["status"] = "EMPTY"
            report["duration"] = timing
        else:
            patched_notebook, status_code, duration = test_result
            if status_code:
                if status_code == -42:
                    status = "TIMEOUT"
                    timeout_notebooks.append(patched_notebook)
                else:
                    status = "FAILED"
                    failed_notebooks.append(patched_notebook)
                report["status"] = status
            else:
                report["status"] = "SUCCESS" if not report["status"] in ["TIMEOUT", "FAILED"] else report["status"]

            timing += duration
            report["duration"] = timing
            if args.collect_reports:
                write_single_notebook_report(patched_notebook, status_code, duration, reports_dir)
            if args.early_stop:
                break

    exit_status = finalize_status(failed_notebooks, timeout_notebooks, test_plan, reports_dir, root)
    return exit_status


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
