import nbformat
import sys
from pathlib import Path


def patch_notebooks(notebooks_dir):
    """
    Patch notebooks in notebooks directory with replacement values
    found in notebook metadata to speed up test execution.
    This function is specific for the OpenVINO notebooks
    Github Actions CI.

    For example: change nr of epochs from 15 to 1 in
    301-tensorflow-training-openvino-pot.ipynb by adding
    {"test_replace": {"epochs = 15": "epochs = 1"} to the cell
    metadata of the cell that contains `epochs = 15`

    :param notebooks_dir: Directory that contains the notebook subdirectories.
                          For example: openvino_notebooks/notebooks
    """
    for notebookfile in Path(notebooks_dir).glob("**/[0-9]*.ipynb"):
        nb = nbformat.read(notebookfile, as_version=nbformat.NO_CONVERT)
        found = False
        for cell in nb["cells"]:
            replace_dict = cell.get("metadata", {}).get("test_replace")
            if replace_dict is not None:
                found = True
                for source_value, target_value in replace_dict.items():
                    if source_value not in cell["source"]:
                        raise ValueError(
                            f"Processing {notebookfile} failed: {source_value} does not exist in cell"
                        )
                    cell["source"] = cell["source"].replace(source_value, target_value)
                    cell["source"] = "# Modified for testing\n" + cell["source"]
                    print(f"Processed {notebookfile}: {source_value} -> {target_value}")
        if not found:
            print(f"No replacements found for {notebookfile}")
        nbformat.write(
            nb,
            notebookfile.with_name(f"test_{notebookfile.name}"),
            version=nbformat.NO_CONVERT,
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If this script is called without any arguments, patch notebooks
        # in the current directory (recursively).
        notebooks_dir = "."
    else:
        # If the script is called with a command line argument, it is expected
        # to be the path to the notebooks directory
        notebooks_dir = sys.argv[1]
        if not Path(notebooks_dir).is_dir():
            raise ValueError(f"'{notebooks_dir}' is not an existing directory")

    patch_notebooks(notebooks_dir)
