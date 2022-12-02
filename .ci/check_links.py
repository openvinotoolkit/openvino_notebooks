#!/usr/bin/env python3

"""
This script is like check-basics.py, but specific to the documentation.
It's split off into a separate script, so that it can be easily run on its own.
"""

import os
import sys
import mistune
import requests
import urllib.parse
import urllib.request

from pathlib import Path

NOTEBOOKS_ROOT = Path(__file__).resolve().parents[1]

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
        # print(message, file=sys.stderr)
        print(message)

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
                file_name = os.path.join(os.path.dirname(md_path), components.path)
                if not os.path.exists(file_name):
                    complain(f'{md_path}: invalid URL reference {url!r}')
                continue

            try:
                get = requests.get(url, proxies={"http": "http://proxy-chain.intel.com:911", 'https': 'http://proxy-chain.intel.com:912'})
                if get.status_code != 200:
                    complain(f'{md_path}: URL can not be reached {url!r}, status code {get.status_code}')    
            except Exception as err:
                complain(f'{md_path}: URL can not be reached {url!r}, error {err}')


if __name__ == '__main__':
    main()