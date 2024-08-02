import argparse
from pathlib import Path
import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_scarf_tag(file_path: Path) -> str:
    if file_path.is_absolute():
        file_path = file_path.relative_to(REPO_ROOT)
    return f'<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file={str(file_path)}" />'


def check_scarf_tag(file_path: Path) -> bool:
    if file_path.suffix == ".ipynb":
        return check_scarf_tag_in_notebook(file_path)
    if file_path.suffix == ".md":
        return check_scarf_tag_in_readme(file_path)
    print(f'Invalid file extension at path "{str(file_path)}". Only .ipynb and .md files are supported.')
    return False


def check_scarf_tag_in_notebook(notebook_path: Path) -> bool:
    expected_scarf_tag = get_scarf_tag(notebook_path)
    with open(notebook_path, "r") as notebook_file:
        nb_node: nbformat.NotebookNode = nbformat.read(notebook_file, as_version=4)
        first_cell_source: str = nb_node["cells"][0]["source"]
        first_cell_source_lines = first_cell_source.split("\n")
        return any([expected_scarf_tag in line for line in first_cell_source_lines])


def check_scarf_tag_in_readme(readme_path: Path) -> bool:
    expected_scarf_tag = get_scarf_tag(readme_path)
    with open(readme_path, "r", encoding="utf8") as readme_file:
        readme_content_lines = readme_file.readlines()
        return any([expected_scarf_tag in line for line in readme_content_lines])


def add_scarf_tag(file_path: Path):
    if file_path.suffix == ".ipynb":
        add_scarf_tag_to_notebook(file_path)
    elif file_path.suffix == ".md":
        add_scarf_tag_to_readme(file_path)
    else:
        raise Exception(f'Invalid file extension at path "{str(file_path)}". Only .ipynb and .md files are supported.')


def add_scarf_tag_to_notebook(notebook_path: Path):
    with open(notebook_path, "r") as fr:
        nb_node: nbformat.NotebookNode = nbformat.read(fr, as_version=4)
        first_cell_source: str = nb_node["cells"][0]["source"]
        first_cell_source_lines = first_cell_source.split("\n")
        first_cell_source_lines.append("")
        first_cell_source_lines.append(get_scarf_tag(notebook_path))
        first_cell_source_lines.append("")
        nb_node["cells"][0]["source"] = "\n".join(first_cell_source_lines)
    with open(notebook_path, "w") as fw:
        nbformat.write(nb_node, fw)


def add_scarf_tag_to_readme(readme_path: Path):
    with open(readme_path, "r", encoding="utf8") as fr:
        content_lines = fr.readlines()
        content_lines.append("\n")
        content_lines.append(get_scarf_tag(readme_path))
        content_lines.append("\n")
    with open(readme_path, "w", encoding="utf8") as fw:
        fw.writelines(content_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--source",
        help="Specify the path to the notebook or README file, where Scarf Pixel tag should be added",
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
    add_scarf_tag(file_path)
