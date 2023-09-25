#!/usr/bin/env python3

import sys
import mistune
import requests
import urllib.parse

from pathlib import Path

NOTEBOOKS_ROOT = Path(__file__).resolve().parents[1]

EXCEPTIONS_URLs = ["medium.com", "https://www.paddlepaddle.org.cn/", "mybinder.org", "https://arxiv.org"]

def get_all_ast_nodes(ast_nodes):
    for node in ast_nodes:
        yield node
        if 'children' in node:
            yield from get_all_ast_nodes(node['children'])

def get_all_references_from_md(md_path):
    parse_markdown = mistune.create_markdown(renderer=mistune.AstRenderer())
    ast = parse_markdown(md_path.read_text(encoding='UTF-8'))

    for node in get_all_ast_nodes(ast):
        if node['type'] == 'image':
            yield node['src']
        elif node['type'] == 'link':
            yield node['link']


def main():
    all_passed = True

    def complain(message):
        nonlocal all_passed
        all_passed = False
        print(message, file=sys.stderr)

    for md_path in NOTEBOOKS_ROOT.glob('**/*README*.md'): 
        for url in get_all_references_from_md(md_path):

            try:
                components = urllib.parse.urlparse(url)
            except ValueError:
                complain(f'{md_path}: invalid URL reference {url!r}')
                continue

            if not components.path: # self-link
                continue

            if not components.scheme and not components.netloc:
                # check if it is relative path on file from repo
                file_name = md_path.parent / components.path
                if not file_name.exists():
                    complain(f'{md_path}: invalid URL reference {url!r}')
                continue

            try:
                get = requests.get(url, timeout=5)
                if get.status_code != 200:
                    if get.status_code in [500, 429, 443] and any([known_url in url for known_url in EXCEPTIONS_URLs]):
                        print(f'{md_path}: URL can not be reached {url!r}, status code {get.status_code}')
                        continue
                    complain(f'{md_path}: URL can not be reached {url!r}, status code {get.status_code}')    
            except Exception as err:
                if any([known_url in url for known_url in EXCEPTIONS_URLs]):
                    print(f'{md_path}: URL can not be reached {url!r}, error {err}')
                else:    
                    complain(f'{md_path}: URL can not be reached {url!r}, error {err}')

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()