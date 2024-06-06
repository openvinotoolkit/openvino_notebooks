import argparse
import re
from pathlib import Path
import nbformat
import nbconvert
from traitlets.config import Config


# Notebooks that are excluded from the CI tests
EXCLUDED_NOTEBOOKS = ["data-preparation-ct-scan.ipynb", "pytorch-monai-training.ipynb"]

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


def disable_skip_ext(nb, notebook_path, test_device=""):
    found = False

    skip_for_device = None if test_device else False
    for cell in nb["cells"]:
        if test_device is not None and skip_for_device is None:
            if (
                'skip_for_device = "{}" in device.value'.format(test_device) in cell["source"]
                and "to_quantize = widgets.Checkbox(value=not skip_for_device" in cell["source"]
            ):
                skip_for_device = True

        if "%%skip" in cell["source"]:
            found = True
            if not skip_for_device:
                cell["source"] = re.sub(r"%%skip.*.\n", "\n", cell["source"])
            else:
                cell["source"] = ""
    if found:
        print(f"Disabled skip extension mode for {notebook_path}")
    return nb


def remove_ov_install(cell):
    updated_lines = []

    def has_additional_deps(str_part):
        if "%pip" in str_part:
            return False
        if "install" in str_part:
            return False
        if str_part.startswith("-"):
            return False
        if str_part.startswith("https://"):
            return False
        return True

    lines = cell["source"].split("\n")
    for line in lines:
        if "openvino" in line:
            if "optimum-cli" in line or line.startswith("#"):
                updated_lines.append(line)
                continue
            updated_line_content = []
            empty = True
            package_found = False
            for part in line.split(" "):
                if "openvino-dev" in part:
                    package_found = True
                    continue
                if "openvino-nightly" in part:
                    package_found = True
                    continue
                if "openvino-tokenizers" in part:
                    package_found = True
                    continue
                if "openvino>" in part or "openvino=" in part or "openvino" == part:
                    package_found = True
                    continue
                if empty:
                    empty = not has_additional_deps(part)
                updated_line_content.append(part)

            if package_found:
                if not empty:
                    updated_line = " ".join(updated_line_content)
                    if line.startswith(" "):
                        for token in line:
                            if token != " ":
                                break
                            # keep indention
                            updated_line = " " + updated_line
                    updated_lines.append(updated_line + "\n# " + line)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)
    cell["source"] = "\n".join(updated_lines)


def patch_notebooks(notebooks_dir, test_device="", skip_ov_install=False):
    """
    Patch notebooks in notebooks directory with replacement values
    found in notebook metadata to speed up test execution.
    This function is specific for the OpenVINO notebooks
    Github Actions CI.

    For example: change nr of epochs from 15 to 1 in
    tensorflow-training-openvino-nncf.ipynb by adding
    {"test_replace": {"epochs = 15": "epochs = 1"} to the cell
    metadata of the cell that contains `epochs = 15`

    :param notebooks_dir: Directory that contains the notebook subdirectories.
                          For example: openvino_notebooks/notebooks
    """

    nb_convert_config = Config()
    nb_convert_config.NotebookExporter.preprocessors = ["nbconvert.preprocessors.ClearOutputPreprocessor"]
    output_remover = nbconvert.NotebookExporter(nb_convert_config)
    for notebookfile in Path(notebooks_dir).glob("**/*.ipynb"):
        if not str(notebookfile.name).startswith("test_") and notebookfile.name not in EXCLUDED_NOTEBOOKS:
            nb = nbformat.read(notebookfile, as_version=nbformat.NO_CONVERT)
            found = False
            device_found = False
            for cell in nb["cells"]:
                if skip_ov_install and "%pip" in cell["source"]:
                    remove_ov_install(cell)
                if test_device and DEVICE_WIDGET in cell["source"]:
                    device_found = True
                    cell["source"] = re.sub(r"value=.*,", f"value='{test_device.upper()}',", cell["source"])
                    cell["source"] = re.sub(
                        r"options=",
                        f"options=['{test_device.upper()}'] + ",
                        cell["source"],
                    )
                    print(f"Replaced testing device to {test_device}")
                replace_dict = cell.get("metadata", {}).get("test_replace")
                if replace_dict is not None:
                    found = True
                    for source_value, target_value in replace_dict.items():
                        if source_value not in cell["source"]:
                            raise ValueError(f"Processing {notebookfile} failed: {source_value} does not exist in cell")
                        cell["source"] = cell["source"].replace(source_value, target_value)
                        cell["source"] = "# Modified for testing\n" + cell["source"]
                        print(f"Processed {notebookfile}: {source_value} -> {target_value}")
            if test_device and not device_found:
                print(f"No device replacement found for {notebookfile}")
            if not found:
                print(f"No replacements found for {notebookfile}")
            disable_gradio_debug(nb, notebookfile)
            disable_skip_ext(nb, notebookfile, args.test_device)
            nb_without_out, _ = output_remover.from_notebook_node(nb)
            with notebookfile.with_name(f"test_{notebookfile.name}").open("w", encoding="utf-8") as out_file:
                out_file.write(nb_without_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Notebook patcher")
    parser.add_argument("notebooks_dir", default=".")
    parser.add_argument("-td", "--test_device", default="")
    parser.add_argument("--skip_ov_install", action="store_true")
    args = parser.parse_args()
    if not Path(args.notebooks_dir).is_dir():
        raise ValueError(f"'{args.notebooks_dir}' is not an existing directory")
    patch_notebooks(args.notebooks_dir, args.test_device, args.skip_ov_install)
