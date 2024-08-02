import argparse
from pathlib import Path
from typing import Optional, Tuple
import nbformat


INSTALL_INSTRUCTIONS_CONTENT = """### Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide)."""


def find_toc_cell(nb_node: nbformat.NotebookNode) -> Tuple[Optional[str], Optional[int]]:
    for i, cell in enumerate(nb_node["cells"]):
        if "#### Table of contents:" in cell["source"]:
            return (cell["source"], i)
    return (None, None)


def check_install_instructions(notebook_path: Path) -> bool:
    with open(notebook_path, "r") as notebook_file:
        nb_node: nbformat.NotebookNode = nbformat.read(notebook_file, as_version=4)
        cell_source, i = find_toc_cell(nb_node)
        if not cell_source:
            print(f'ToC is not found in notebook "{str(notebook_path)}"')
            return False
        return INSTALL_INSTRUCTIONS_CONTENT in cell_source


def add_install_instructions(notebook_path: Path):
    with open(notebook_path, "r") as fr:
        nb_node: nbformat.NotebookNode = nbformat.read(fr, as_version=4)
        cell_source, i = find_toc_cell(nb_node)
        if not cell_source:
            print(f'ToC is not found in notebook "{str(notebook_path)}"')
            return
        cell_source_lines = cell_source.split("\n")
        cell_source_lines.append(INSTALL_INSTRUCTIONS_CONTENT)
        nb_node["cells"][i]["source"] = "\n".join(cell_source_lines)
    with open(notebook_path, "w") as fw:
        nbformat.write(nb_node, fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--source",
        help="Specify the path to the notebook file, where Install Instructions section should be added",
        required=True,
    )

    args = parser.parse_args()
    file_path = Path(args.source)
    if not file_path.exists():
        print(f'File does not exist at path "{file_path}"')
        exit(1)
    if not file_path.is_file():
        print(f"Provided path is not a file")
        exit(1)
    add_install_instructions(file_path)
