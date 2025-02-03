import argparse
from pathlib import Path
import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]


def _get_telemetry_snippet(notebook_path: Path) -> str:
    if notebook_path.is_absolute():
        notebook_path = notebook_path.relative_to(REPO_ROOT)
    return "".join(
        [
            "# Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry\n",
            "from notebook_utils import collect_telemetry\n\n",
            f'collect_telemetry("{notebook_path.name}")',
        ]
    )


def check_telemetry_snippet(notebook_path: Path) -> bool:
    if notebook_path.suffix != ".ipynb":
        print(f'Invalid file extension at path "{str(notebook_path)}". Only .ipynb files are supported.')
        return False
    telemetry_snippet = _get_telemetry_snippet(notebook_path)
    with open(notebook_path, "r") as notebook_file:
        nb_node: nbformat.NotebookNode = nbformat.read(notebook_file, as_version=4)
    for cell in nb_node["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell_content: str = cell["source"]
        if telemetry_snippet in cell_content:
            return True
    return False


def _add_telemetry_snippet(notebook_path: Path):
    if notebook_path.suffix != ".ipynb":
        raise Exception(f'Invalid file extension at path "{str(notebook_path)}". Only .ipynb files are supported.')
    with open(notebook_path, "r") as fr:
        nb_node: nbformat.NotebookNode = nbformat.read(fr, as_version=4)
        # Find cell with notebook_utils
        notebook_utils_url = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py"
        for i, cell in enumerate(nb_node["cells"]):
            if cell["cell_type"] != "code":
                continue
            cell_content: str = cell["source"]
            if notebook_utils_url in cell_content:
                nb_node["cells"][i]["source"] = cell_content + "\n\n" + _get_telemetry_snippet(notebook_path)
                break
    with open(notebook_path, "w") as fw:
        nbformat.write(nb_node, fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--source",
        help="Specify the path to the notebook file, where telemetry snippet should be added",
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
    _add_telemetry_snippet(file_path)
