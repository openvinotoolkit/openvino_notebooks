import json
import pathlib
import argparse
import re

TABLE_OF_CONTENT = r'#+\s+Table of content:?'


def find_tc_in_cell(cell):
    tc_cell = None
    tc_line_number = None
    for i, line in enumerate(cell["source"]):
        if re.match(TABLE_OF_CONTENT, line):
            tc_cell = cell
            tc_line_number = i
            break

    return tc_cell, tc_line_number


def create_title_for_tc(title):
    title_for_tc = title.lstrip("#").lstrip()
    title_for_tc = re.sub(r'[\[\]\n]', '', title_for_tc)
    title_for_tc = re.sub(r'\(http.*\)', '', title_for_tc)

    return title_for_tc


def create_link_for_tc(title):
    link = re.sub(r'[`$^]', '', title)
    link = link.replace(" ", "-")
    link += '-Uparrow'

    return link


def remove_old_tc(cell, idx):
    if cell is not None:
        for line in cell["source"][idx:]:
            if re.match(r'\s*-\s*\[.*\]\(#.*\).*', line) or\
               re.match(TABLE_OF_CONTENT, line):
                cell["source"].remove(line)
    return cell


def get_tc_line(title, title_for_tc, link):
    # calc indents for Table of content
    indents = ("  " * ((title.index(" ") - 2) * 2)) + "-" + " "
    line = f"{indents}[{title_for_tc}](#{link})\n"

    return line


def is_markdown(cell):
    return "markdown" == cell["cell_type"]


def is_title(line):
    return line.strip().startswith("#") and line.strip().lstrip("#").lstrip()


def generate_table_of_content(notebook_path: pathlib.Path):
    table_of_content = ["\n", "### Table of content:\n"]

    table_of_content_cell = None
    table_of_content_cell_idx = None

    with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
        notebook_json = json.load(notebook_file)

    table_of_content_cell, table_of_content_cell_idx =\
        find_tc_in_cell(notebook_json["cells"][0])

    for cell in filter(is_markdown, notebook_json["cells"][1:]):
        if table_of_content_cell is None:
            table_of_content_cell, table_of_content_cell_idx = find_tc_in_cell(cell)

        titles = [line for line in cell["source"] if is_title(line)]
        for title in titles:
            idx = cell["source"].index(title)

            title = title.strip()
            if '[$\\Uparrow$](#Table-of-content:)' in cell["source"][idx]:
                title = title.replace(" [$\\Uparrow$](#Table-of-content:)", "")
            else:
                cell["source"][idx] = title + ' [$\\Uparrow$](#Table-of-content:)\n'

            title_for_tc = create_title_for_tc(title)
            link_for_tc = create_link_for_tc(title_for_tc)
            new_line = get_tc_line(title, title_for_tc, link_for_tc)
            table_of_content.append(new_line)

    if table_of_content_cell is not None:
        table_of_content_cell = remove_old_tc(table_of_content_cell, table_of_content_cell_idx)

    if table_of_content_cell is not None:
        table_of_content_cell["source"].extend(table_of_content)
    else:
        notebook_json["cells"][0]["source"].extend(table_of_content)

    with open(notebook_path, "w", encoding='utf-8') as in_f:
        json.dump(notebook_json, in_f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source',
                        help='Please, specify notebook or folder with notebooks.\
                            Table of content will be added or modified in each.')

    args = parser.parse_args()
    path_to_source = pathlib.Path(args.source)
    if not path_to_source.exists():
        print(f'Incorrect path to notebook(s) {path_to_source}')
        exit()
    elif path_to_source.is_file():
        generate_table_of_content(path_to_source)
    elif path_to_source.is_dir():
        for notebook in path_to_source.glob('**/*.ipynb'):
            generate_table_of_content(notebook)
