import json
import glob
import mmap
import sys

def get_notebooks(path: str):
    return glob.glob(f"{path}/*/[0-9]*.ipynb")

def get_tags(path: str):
    return json.load(open(path))

def find_tags_for_notebook(notebook_path: str, tags: dict):
    nb_tags = []
    with open(notebook_path) as file:
        f = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        for tag, keywords in tags.items():
            for keyword in keywords:
                if f.find(bytes(keyword, 'utf-8')) != -1:
                    nb_tags.append(tag)
                    break
    return nb_tags

def find_tags_for_all_notebooks(notebooks: list, tags: dict):
    notebooks_tags = {}
    for notebook in notebooks:
        nb_tags = sorted(find_tags_for_notebook(notebook, tags))
        if nb_tags: 
            notebooks_tags[notebook.split('/')[-1].split('.')[0]] = nb_tags
    return notebooks_tags

if __name__ == "__main__":
    if len(sys.argv) == 1:
        notebooks_paths = sorted(get_notebooks("notebooks"))
        tags = get_tags(".ci/keywords.json")['tags']
    else:
        notebooks_paths = sorted(get_notebooks('/'.join(sys.argv[1].split('/')[:-2])))
        tags = get_tags(sys.argv[2])['tags']
    all_notebooks_tags = find_tags_for_all_notebooks(notebooks_paths, tags)
    print(json.dumps(all_notebooks_tags, indent=4))