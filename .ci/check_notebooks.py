import sys
import json
from table_of_content import find_tc_in_cell
from patch_notebooks import DEVICE_WIDGET
from pathlib import Path

NOTEBOOKS_ROOT = Path(__file__).resolve().parents[1]


def main():
    all_passed = True
    no_tocs = []
    no_device = []

    def complain(message):
        nonlocal all_passed
        all_passed = False
        print(message, file=sys.stderr)

    for nb_path in NOTEBOOKS_ROOT.glob("notebooks/**/*.ipynb"):
        with open(nb_path, "r", encoding="utf-8") as notebook_file:
            notebook_json = json.load(notebook_file)
            toc_found = False
            device_found = False
            for cell in notebook_json["cells"]:

                if not toc_found and cell["cell_type"] == "markdown":
                    tc_cell, tc_line = find_tc_in_cell(cell)
                    if tc_line is not None:
                        toc_found = True

                if not device_found and cell["cell_type"] == "code":
                    device_found = DEVICE_WIDGET in cell["source"]

                if toc_found and device_found:
                    break
            if not toc_found:
                no_tocs.append(nb_path.relative_to(NOTEBOOKS_ROOT))
                complain(f"FAILED: {nb_path.relative_to(NOTEBOOKS_ROOT)}: table of content is not found")
            if not device_found:
                no_device.append(nb_path.relative_to(NOTEBOOKS_ROOT))
                complain(f"FAILEd: {nb_path.relative_to(NOTEBOOKS_ROOT)}: device widget is not found")

    if not all_passed:
        print("SUMMARY:")
        print("==================================")
        print("NO TABLE OF CONTENT:")
        print("\n".join(no_tocs))
        print("==================================")
        print("NO DEVICE SELECTION:")
        print("\n".join(no_device))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
