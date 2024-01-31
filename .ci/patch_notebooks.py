import argparse
import re
from pathlib import Path
import nbformat
import nbconvert
from traitlets.config import Config


# Notebooks that are excluded from the CI tests
EXCLUDED_NOTEBOOKS = ["data-preparation-ct-scan.ipynb"]

DEVICE_WIDGET = "device = widgets.Dropdown("

def disable_gradio_debug(nb, notebook_path):
    found = False
    for cell in nb["cells"]:
        if "gradio" in cell["source"] and "debug" in cell["source"]:
            found = True
            cell["source"] = cell["source"].replace("debug=True", "debug=False")
    
    if found:
        print(f"Disabled gradio debug mode for {notebook_path}")
    return nb


def disable_skip_ext(nb, notebook_path):
    found = False
    for cell in nb["cells"]:
        if "%%skip" in cell["source"]:
            found = True
            cell["source"] = re.sub(r"%%skip.*.\n", "\n", cell["source"])
    if found:
        print(f"Disabled skip extension mode for {notebook_path}")
    return nb


def patch_notebooks(notebooks_dir, test_device=""):
    """
    Patch notebooks in notebooks directory with replacement values
    found in notebook metadata to speed up test execution.
    This function is specific for the OpenVINO notebooks
    Github Actions CI.

    For example: change nr of epochs from 15 to 1 in
    301-tensorflow-training-openvino-nncf.ipynb by adding
    {"test_replace": {"epochs = 15": "epochs = 1"} to the cell
    metadata of the cell that contains `epochs = 15`

    :param notebooks_dir: Directory that contains the notebook subdirectories.
                          For example: openvino_notebooks/notebooks
    """

    nb_convert_config = Config()
    nb_convert_config.NotebookExporter.preprocessors = ["nbconvert.preprocessors.ClearOutputPreprocessor"]
    output_remover = nbconvert.NotebookExporter(nb_convert_config)
    for notebookfile in Path(notebooks_dir).glob("**/*.ipynb"):
        if (
            not str(notebookfile.name).startswith("test_")
            and notebookfile.name not in EXCLUDED_NOTEBOOKS
        ):
            nb = nbformat.read(notebookfile, as_version=nbformat.NO_CONVERT)
            found = False
            device_found = False
            for cell in nb["cells"]:
                if test_device and DEVICE_WIDGET in cell["source"]:
                    device_found = True
                    cell["source"] = re.sub(r"value=.*,", f"value='{test_device.upper()}',", cell["source"])
                    cell["source"] = re.sub(r"options=.*,", f"options=['{test_device.upper()}'],", cell["source"])
                    print(f"Replaced testing device to {test_device}")
                replace_dict = cell.get("metadata", {}).get("test_replace")
                if replace_dict is not None:
                    found = True
                    for source_value, target_value in replace_dict.items():
                        if source_value not in cell["source"]:
                            raise ValueError(
                                f"Processing {notebookfile} failed: {source_value} does not exist in cell"
                            )
                        cell["source"] = cell["source"].replace(
                            source_value, target_value
                        )
                        cell["source"] = "# Modified for testing\n" + cell["source"]
                        print(
                            f"Processed {notebookfile}: {source_value} -> {target_value}"
                        )
            if test_device and not device_found:
                print(f"No device replacement found for {notebookfile}")
            if not found:
                print(f"No replacements found for {notebookfile}")
            disable_gradio_debug(nb, notebookfile)
            disable_skip_ext(nb, notebookfile)
            nb_without_out, _ = output_remover.from_notebook_node(nb)
            with notebookfile.with_name(f"test_{notebookfile.name}").open("w", encoding="utf-8") as out_file:
                out_file.write(nb_without_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Notebook patcher")
    parser.add_argument("notebooks_dir", default=".")
    parser.add_argument("-td", "--test_device", default="")
    args = parser.parse_args()
    if not Path(args.notebooks_dir).is_dir():
        raise ValueError(f"'{args.notebooks_dir}' is not an existing directory")
    patch_notebooks(args.notebooks_dir, args.test_device)
