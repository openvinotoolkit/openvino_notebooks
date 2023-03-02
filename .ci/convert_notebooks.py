import argparse
from pathlib import Path
import time
import subprocess
import shutil


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude_execution_file")
    parser.add_argument("--exclude_conversion_file")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="timeout for notebook execution")
    parser.add_argument("--rst_dir", type=Path,
                        help="rst files output directory", default=Path("rst"))
    parser.add_argument("--html_dir", type=Path,
                        help="html files output directory", default=Path("html"))
    parser.add_argument("--markdown_dir", type=Path,
                        help="markdown files output directory", default=Path("markdown"))

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
    markdown_failed = []
    rst_failed = []
    html_failed = []
    if args.exclude_conversion_file is not None:
        ignore_conversion_list = prepare_ignore_list(args.exclude_conversion_file)
    if args.exclude_execution_file is not None:
        ignore_execution_list = prepare_ignore_list(args.exclude_execution_file)
    ROOT = Path(__file__).parents[1]
    notebooks_dir = ROOT / "notebooks"
    notebooks = sorted(list(notebooks_dir.rglob('**/*.ipynb')))
    for notebook in notebooks:
        notebook_path = notebook.relative_to(ROOT)
        if str(notebook_path) in ignore_conversion_list:
            continue
        notebook_executed = notebook_path.parent / notebook_path.name.replace(".ipynb", "-with-output.ipynb")
        start = time.perf_counter()
        print(f"Convert {notebook_path}")
        if str(notebook_path) not in ignore_execution_list:
            retcode = subprocess.run(["jupyter", "nbconvert",  "--log-level=INFO", "--execute", "--to",  "notebook", "--output",
                                     str(notebook_executed),  '--output-dir', str(ROOT), '--ExecutePreprocessor.kernel_name=python3', str(notebook_path)], timeout=args.timeout).returncode
            if retcode:
                failed_notebooks.append(str(notebook_path))
                continue
        else:
            shutil.copyfile(notebook_path, notebook_executed)
        markdown_retcode = subprocess.run(["jupyter", "nbconvert", "--to", "markdown", str(notebook_executed), "--output-dir", str(args.markdown_dir),
                                          "--TagRemovePreprocessor.remove_all_outputs_tags=hide_output --TagRemovePreprocessor.enabled=True"], timeout=args.timeout).returncode

        html_retcode = subprocess.run(["jupyter", "nbconvert", "--to", "html", str(notebook_executed), "--output-dir", str(args.html_dir),
                                          "--TagRemovePreprocessor.remove_all_outputs_tags=hide_output --TagRemovePreprocessor.enabled=True"], timeout=args.timeout).returncode
        rst_retcode = subprocess.run(["jupyter", "nbconvert", "--to", "rst", str(notebook_executed), "--output-dir", str(args.rst_dir),
                                          "--TagRemovePreprocessor.remove_all_outputs_tags=hide_output --TagRemovePreprocessor.enabled=True"], timeout=args.timeout).returncode
        end = time.perf_counter() - start
        print(f"Notebook conversion took: {end:.4f} s")
        if markdown_retcode:
            markdown_failed.append(str(notebook_path))
        if html_retcode:
            html_failed.append(str(notebook_path))
        if rst_retcode:
            rst_failed.append(str(notebook_path))
    
    if failed_notebooks:
        print("EXECUTION FAILED:")
        print("\n".join(failed_notebooks))
    if markdown_failed:
        print("MARKDOWN CONVERSION FAILED:")
        print("\n".join(markdown_failed))

    if html_failed:
        print("HTML CONVERSION FAILED:")
        print("\n".join(html_failed))

    if rst_failed:
        print("RST CONVERSION FAILED:")
        print("\n".join(rst_failed))


if __name__ == "__main__":
    main()