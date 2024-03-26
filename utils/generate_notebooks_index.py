import json
from pathlib import Path
from typing import List, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
INDEX_FILE_NAME = "index.md"


def get_skipped_notebooks() -> List[str]:
    skipped_notebooks_file_path = (
        REPO_ROOT / "selector" / "src" / "notebook-metadata" / "skipped-notebooks.txt"
    )
    with open(skipped_notebooks_file_path, "r", encoding="utf-8") as file:
        return list(map(lambda v: v.strip(), file.readlines()))


def get_notebooks_paths() -> List[Path]:
    all_files = NOTEBOOKS_DIR.glob("**/*.ipynb")
    ignored_files = list(NOTEBOOKS_DIR.glob("**/.ipynb_checkpoints/*"))
    ignored_files.extend(NOTEBOOKS_DIR.glob("**/notebook_utils.ipynb"))
    for skipped_notebook in get_skipped_notebooks():
        ignored_files.extend(NOTEBOOKS_DIR.glob(skipped_notebook))
    return sorted(set(all_files) - set(ignored_files))


def get_notebook_tags(notebook_path: Path) -> Dict[str, List[str]]:
    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook_json = json.load(notebook_file)
    return notebook_json["metadata"]["openvino_notebooks"]["tags"]


def write_to_file(file_path: Path, content: str):
    with (file_path).open("w") as file:
        file.write(content)


def collect_notebooks_category_to_paths_map() -> Dict[str, List[Path]]:
    result_map = {}
    for notebook_path in get_notebooks_paths():
        categories_tags = get_notebook_tags(notebook_path)["categories"]
        relative_notebook_path = notebook_path.relative_to(NOTEBOOKS_DIR)
        for category in categories_tags:
            if category in result_map:
                result_map[category].append(relative_notebook_path)
            else:
                result_map[category] = [relative_notebook_path]
    return result_map


def format_to_index_markdown(
    notebooks_category_to_paths_map: Dict[str, List[Path]]
) -> str:
    md_contents = ["# OpenVINO Notebooks - Categories\n\n"]
    for category, notebooks_paths in sorted(notebooks_category_to_paths_map.items()):
        md_contents.append(f"## {category}\n\n")
        for notebook_path in notebooks_paths:
            md_contents.append(f"- [{str(notebook_path)}](./{str(notebook_path)})\n")
        md_contents.append("\n")
    return "".join(md_contents)


def main():
    notebooks_category_to_paths_map = collect_notebooks_category_to_paths_map()
    index_md_content = format_to_index_markdown(notebooks_category_to_paths_map)
    write_to_file(NOTEBOOKS_DIR / INDEX_FILE_NAME, index_md_content)


if __name__ == "__main__":
    main()
