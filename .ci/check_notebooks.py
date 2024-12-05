import sys
import json
from table_of_content import find_tc_in_cell
from patch_notebooks import DEVICE_WIDGET, DEVICE_WIDGET_NEW
from install_instructions import check_install_instructions
from scarf_pixel import check_scarf_tag
from pathlib import Path

NOTEBOOKS_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_NO_DEVICE = [
    Path("notebooks/auto-device/auto-device.ipynb"),  # auto device expected to be used
    Path("notebooks/convert-to-openvino/convert-to-openvino.ipynb"),  # device-agnostic
    Path("notebooks/gpu-device/gpu-device.ipynb"),  # gpu device expected to be used
    Path("notebooks/hello-npu/hello-npu.ipynb"),  # npu device expected to be used
    Path("notebooks/model-server/model-server.ipynb"),  # can not change device in docker configuration on the fly
    Path("notebooks/openvino-tokenizers/openvino-tokenizers.ipynb"),  # cpu required for loading extensions
    Path("notebooks/sparsity-optimization/sparsity-optimization.ipynb"),  # cpu expected to be used
    Path("notebooks/s3d-mil-nce-text-to-video-retrieval/s3d-mil-nce-text-to-video-retrieval.ipynb"),  # only cpu
    Path("notebooks/explainable-ai-2-deep-dive/explainable-ai-2-deep-dive.ipynb"),  # device-agnostic
    Path("notebooks/localai/localai.ipynb"),  # can not change device in docker configuration on the fly
]


def find_device_in_cell(cell):
    for line_idx, line in enumerate(cell["source"]):
        if DEVICE_WIDGET in line or DEVICE_WIDGET_NEW in line:
            return line_idx
    return None


def main():
    all_passed = True
    no_tocs = []
    no_device = []
    no_scarf_tag = []
    no_install_instructions = []

    def complain(message):
        nonlocal all_passed
        all_passed = False
        print(message, file=sys.stderr)

    checkpoints_paths = set(NOTEBOOKS_ROOT.glob("**/.ipynb_checkpoints/*"))

    for nb_path in NOTEBOOKS_ROOT.glob("notebooks/**/*.ipynb"):
        if nb_path in checkpoints_paths:
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
            if not check_install_instructions(nb_path):
                no_install_instructions.append(str(nb_path.relative_to(NOTEBOOKS_ROOT)))
                complain(f"FAILED: {nb_path.relative_to(NOTEBOOKS_ROOT)}: Install Instructions section is not found")

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
        if no_install_instructions:
            print("NO INSTALL INSTRUCTIONS SECTION:")
            print("\n".join(no_install_instructions))
            print("\nYou can generate Install Instructions with the following command:\n    python .ci/install_instructions.py -s <PATH>")
            print("==================================")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
