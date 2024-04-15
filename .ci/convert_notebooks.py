import argparse
import shutil
import subprocess  # nosec - disable B404:import-subprocess check
import time
from pathlib import Path
import nbformat


def disable_gradio_debug(notebook_path):
    nb = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)
    found = False
    for cell in nb["cells"]:
        if "gradio" in cell["source"] and "debug" in cell["source"]:
            found = True
            cell["source"] = cell["source"].replace("debug=True", "debug=False")

    if found:
        print(f"Disabled gradio debug mode for {notebook_path}")
        nbformat.write(nb, str(notebook_path), version=nbformat.NO_CONVERT)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude_execution_file")
    parser.add_argument("--exclude_conversion_file")
    parser.add_argument("--timeout", type=float, default=7200, help="timeout for notebook execution")
    parser.add_argument("--rst_dir", type=Path, help="rst files output directory", default=Path("rst"))

    return parser.parse_args()


def prepare_ignore_list(input_file):
    with Path(input_file).open("r") as f:
        lines = f.readlines()
    return list(map(str.strip, lines))


def main():
    args = arguments()
    ignore_conversion_list = []
    ignore_execution_list = []
    failed_notebooks = []
    rst_failed = []
    if args.exclude_conversion_file is not None:
        ignore_conversion_list = prepare_ignore_list(args.exclude_conversion_file)
    if args.exclude_execution_file is not None:
        ignore_execution_list = prepare_ignore_list(args.exclude_execution_file)
    root = Path(__file__).parents[1]
    notebooks_dir = root / "notebooks"
    notebooks = sorted(list(notebooks_dir.rglob("**/*.ipynb")))
    for notebook in notebooks:
        notebook_path = notebook.relative_to(root)
        if str(notebook_path) in ignore_conversion_list:
            continue
        disable_gradio_debug(notebook_path)
        notebook_executed = notebook_path.parent / notebook_path.name.replace(".ipynb", "-with-output.ipynb")
        start = time.perf_counter()
        print(f"Convert {notebook_path}")
        if str(notebook_path) not in ignore_execution_list:
            try:
                retcode = subprocess.run(
                    [
                        "jupyter",
                        "nbconvert",
                        "--log-level=INFO",
                        "--execute",
                        "--to",
                        "notebook",
                        "--output",
                        str(notebook_executed),
                        "--output-dir",
                        str(root),
                        "--ExecutePreprocessor.kernel_name=python3",
                        str(notebook_path),
                    ],
                    timeout=args.timeout,
                ).returncode
            except subprocess.TimeoutExpired:
                retcode = -42
                print(f"TIMEOUT: {notebook_path}")
            if retcode:
                failed_notebooks.append(str(notebook_path))
                continue
        else:
            shutil.copyfile(notebook_path, notebook_executed)
        rst_retcode = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "rst",
                str(notebook_executed),
                "--output-dir",
                str(args.rst_dir),
                "--TagRemovePreprocessor.remove_all_outputs_tags=hide_output --TagRemovePreprocessor.enabled=True",
            ],
            timeout=args.timeout,
        ).returncode
        notebook_rst = args.rst_dir / notebook_executed.name.replace(".ipynb", ".rst")
        # remove all non-printable characters
        subprocess.run(
            [
                "sed",
                "-i",
                "-e",
                "s/\x1b\[[0-9;]*m//g",
                "-e",
                "s/\x1b\[?25h//g",
                "-e",
                "s/\x1b\[?25l//g",
                str(notebook_rst),
            ],
            timeout=args.timeout,
        )

        end = time.perf_counter() - start
        print(f"Notebook conversion took: {end:.4f} s")
        if rst_retcode:
            rst_failed.append(str(notebook_path))

    if failed_notebooks:
        print("EXECUTION FAILED:")
        print("\n".join(failed_notebooks))

    if rst_failed:
        print("RST CONVERSION FAILED:")
        print("\n".join(rst_failed))


if __name__ == "__main__":
    main()
