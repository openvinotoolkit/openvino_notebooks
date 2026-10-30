import ast
import re
import subprocess  # nosec B404 - fixed git arguments, shell disabled
import sys
import json
from table_of_content import find_tc_in_cell
from patch_notebooks import DEVICE_WIDGET, DEVICE_WIDGET_NEW
from install_instructions import check_install_instructions
from scarf_pixel import check_scarf_tag
from telemetry_snippet import check_telemetry_snippet
from pathlib import Path

NOTEBOOKS_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = NOTEBOOKS_ROOT / "utils"
HELPER_DOWNLOAD_RE = re.compile(r"openvino_notebooks/\w+/utils/|\.\./\.\./utils")

EXPECTED_NO_DEVICE = [
    Path("notebooks/auto-device/auto-device.ipynb"),  # auto device expected to be used
    Path("notebooks/convert-to-openvino/convert-to-openvino.ipynb"),  # device-agnostic
    Path("notebooks/gpu-device/gpu-device.ipynb"),  # gpu device expected to be used
    Path("notebooks/hello-npu/hello-npu.ipynb"),  # npu device expected to be used
    Path("notebooks/model-server/model-server.ipynb"),  # can not change device in docker configuration on the fly
    Path("notebooks/openvino-tokenizers/openvino-tokenizers.ipynb"),  # cpu required for loading extensions
    Path("notebooks/explainable-ai-2-deep-dive/explainable-ai-2-deep-dive.ipynb"),  # device-agnostic
    Path("notebooks/physical-ai-robotics/physical-ai-robotics.ipynb"),  # reference notebook, no inference performed here
]

EXPECTED_NO_TELEMETRY = [
    Path("notebooks/physical-ai-robotics/physical-ai-robotics.ipynb"),  # reference notebook, contains no runnable code
]

EXPECTED_NO_INSTALL = [
    Path("notebooks/physical-ai-robotics/physical-ai-robotics.ipynb"),  # reference notebook, nothing to install or run
]


def find_device_in_cell(cell):
    for line_idx, line in enumerate(cell["source"]):
        if DEVICE_WIDGET in line or DEVICE_WIDGET_NEW in line:
            return line_idx
    return None


def expected_venv_setup_call(nb_path: Path) -> str:
    # Notebooks sharing a directory need separate environments, which requires passing the notebook name
    notebooks_in_dir = [p for p in nb_path.parent.glob("*.ipynb") if not p.name.startswith("test_")]
    return f'setup_notebook_venv("{nb_path.name}")' if len(notebooks_in_dir) > 1 else "setup_notebook_venv()"


def has_venv_setup(nb_path: Path, notebook_json) -> bool:
    code_cells = [cell for cell in notebook_json["cells"] if cell["cell_type"] == "code"]
    return bool(code_cells) and expected_venv_setup_call(nb_path) in "".join(code_cells[0]["source"])


def tracked_files() -> set:
    try:
        output = subprocess.run(["git", "ls-files", "notebooks"], cwd=NOTEBOOKS_ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {NOTEBOOKS_ROOT / line for line in output.splitlines()}


def helper_files_issues(nb_path: Path, notebook_json, tracked: set) -> list:
    """Check that the first code cell lists every `utils` helper the notebook needs and nothing else fetches them."""
    sources = ["".join(cell["source"]) for cell in notebook_json["cells"] if cell["cell_type"] == "code"]
    if not sources:
        return []
    match = re.search(r"^helper_files = (\[.*?\])$", sources[0], re.M)
    if not match:
        return ["first code cell does not define `helper_files`"]
    listed = ast.literal_eval(match.group(1))
    available = {path.name for path in UTILS_DIR.glob("*.py")}
    # helpers with the same name as a file tracked next to the notebook must not overwrite it
    local = {name for name in available if nb_path.parent / name in tracked}
    issues = [f"`helper_files` lists {name}, which is not in utils/" for name in listed if name not in available]
    issues += [f"`helper_files` lists {name}, which would overwrite the notebook's own {name}" for name in listed if name in local]
    code = "\n".join(sources)
    # notebook-local modules that the notebook imports may themselves import helpers
    for module_path in nb_path.parent.glob("*.py"):
        if module_path in tracked and module_path.name not in available and re.search(rf"^\s*(from|import) {module_path.stem}\b", code, re.M):
            code += "\n" + module_path.read_text(encoding="utf-8")
    for name in sorted(available - local - set(listed)):
        module = name[:-3]
        if re.search(rf"^\s*(from {module} import|import {module}\b|%load_ext {module}\b)", code, re.M):
            issues.append(f"uses {module} but `helper_files` does not list {name}")
    for idx, source in enumerate(sources[1:], start=2):
        if HELPER_DOWNLOAD_RE.search(source):
            issues.append(f"code cell {idx} fetches utils helpers, list them in `helper_files` instead")
    return issues


def main():
    all_passed = True
    no_tocs = []
    no_device = []
    no_scarf_tag = []
    no_telemetry_snippet = []
    no_install_instructions = []
    no_venv_setup = []
    bad_helper_files = []
    tracked = tracked_files()

    def complain(message):
        nonlocal all_passed
        all_passed = False
        print(message, file=sys.stderr)

    checkpoints_paths = set(NOTEBOOKS_ROOT.glob("**/.ipynb_checkpoints/*"))

    for nb_path in NOTEBOOKS_ROOT.glob("notebooks/**/*.ipynb"):
        if nb_path in checkpoints_paths or ".venv" in nb_path.parts:
            continue
        with open(nb_path, "r", encoding="utf-8") as notebook_file:
            notebook_json = json.load(notebook_file)
            toc_found = False
            device_found = False
            if nb_path.relative_to(NOTEBOOKS_ROOT) in EXPECTED_NO_DEVICE:
                print(f"SKIPPED: {nb_path.relative_to(NOTEBOOKS_ROOT)} for device wdget check")
                device_found = True
            for cell in notebook_json["cells"]:
                if not toc_found and cell["cell_type"] == "markdown":
                    tc_cell, tc_line = find_tc_in_cell(cell)
                    if tc_line is not None:
                        toc_found = True

                if not device_found and find_device_in_cell(cell) is not None:
                    device_found = True

                if toc_found and device_found:
                    break
            if not toc_found:
                no_tocs.append(str(nb_path.relative_to(NOTEBOOKS_ROOT)))
                complain(f"FAILED: {nb_path.relative_to(NOTEBOOKS_ROOT)}: table of content is not found")
            if not device_found:
                no_device.append(str(nb_path.relative_to(NOTEBOOKS_ROOT)))
                complain(f"FAILED: {nb_path.relative_to(NOTEBOOKS_ROOT)}: device widget is not found")
            if not check_scarf_tag(nb_path):
                no_scarf_tag.append(str(nb_path.relative_to(NOTEBOOKS_ROOT)))
                complain(f"FAILED: {nb_path.relative_to(NOTEBOOKS_ROOT)}: Scarf Pixel tag is not found")
            if nb_path.relative_to(NOTEBOOKS_ROOT) in EXPECTED_NO_TELEMETRY:
                print(f"SKIPPED: {nb_path.relative_to(NOTEBOOKS_ROOT)} for telemetry snippet check")
            elif not check_telemetry_snippet(nb_path):
                no_telemetry_snippet.append(str(nb_path.relative_to(NOTEBOOKS_ROOT)))
                complain(f"FAILED: {nb_path.relative_to(NOTEBOOKS_ROOT)}: telemetry snippet is not found")
            if nb_path.relative_to(NOTEBOOKS_ROOT) in EXPECTED_NO_INSTALL:
                print(f"SKIPPED: {nb_path.relative_to(NOTEBOOKS_ROOT)} for install instructions check")
            elif not check_install_instructions(nb_path):
                no_install_instructions.append(str(nb_path.relative_to(NOTEBOOKS_ROOT)))
                complain(f"FAILED: {nb_path.relative_to(NOTEBOOKS_ROOT)}: Install Instructions section is not found")
            if nb_path.relative_to(NOTEBOOKS_ROOT) not in EXPECTED_NO_INSTALL and not has_venv_setup(nb_path, notebook_json):
                no_venv_setup.append(str(nb_path.relative_to(NOTEBOOKS_ROOT)))
                complain(f"FAILED: {nb_path.relative_to(NOTEBOOKS_ROOT)}: first code cell does not call {expected_venv_setup_call(nb_path)}")
            if nb_path.relative_to(NOTEBOOKS_ROOT) not in EXPECTED_NO_INSTALL:
                for issue in helper_files_issues(nb_path, notebook_json, tracked):
                    bad_helper_files.append(str(nb_path.relative_to(NOTEBOOKS_ROOT)))
                    complain(f"FAILED: {nb_path.relative_to(NOTEBOOKS_ROOT)}: {issue}")

    if not all_passed:
        print("\nSUMMARY:")
        print("==================================")
        if no_tocs:
            print("NO TABLE OF CONTENT:")
            print("\n".join(no_tocs))
            print("\nYou can generate Table of content with the following command:\n    python .ci/table_of_content.py -s <PATH>")
            print("==================================")
        if no_device:
            print("NO DEVICE SELECTION:")
            print("\n".join(no_device))
            print("==================================")
        if no_scarf_tag:
            print("NO SCARF PIXEL TAG:")
            print("\n".join(no_scarf_tag))
            print("\nYou can generate Scarf Pixel tag with the following command:\n    python .ci/scarf_pixel.py -s <PATH>")
            print("==================================")
        if no_telemetry_snippet:
            print("NO TELEMETRY SNIPPET:")
            print("\n".join(no_telemetry_snippet))
            print("\nYou can generate telemetry snippet with the following command:\n    python .ci/telemetry_snippet.py -s <PATH>")
            print("==================================")
        if no_install_instructions:
            print("NO INSTALL INSTRUCTIONS SECTION:")
            print("\n".join(no_install_instructions))
            print("\nYou can generate Install Instructions with the following command:\n    python .ci/install_instructions.py -s <PATH>")
            print("==================================")
        if no_venv_setup:
            print("NO NOTEBOOK VIRTUAL ENVIRONMENT SETUP:")
            print("\n".join(no_venv_setup))
            print("\nThe first code cell should fetch the helper files and call setup_notebook_venv(), see CONTRIBUTING.md")
            print("==================================")
        if bad_helper_files:
            print("INCORRECT HELPER FILES SETUP:")
            print("\n".join(sorted(set(bad_helper_files))))
            print("\nList every helper from utils/ the notebook uses in `helper_files` of the first code cell, see CONTRIBUTING.md")
            print("==================================")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
