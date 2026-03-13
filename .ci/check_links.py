#!/usr/bin/env python3

import argparse
import sys
import time
import mistune
import requests
import urllib.parse
from collections import defaultdict

from pathlib import Path

NOTEBOOKS_ROOT = Path(__file__).resolve().parents[1]

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubles each retry

EXCEPTIONS_URLs = [
    "medium.com",
    "https://www.paddlepaddle.org.cn/",
    "mybinder.org",
    "https://arxiv.org",
    "http://host.robots.ox.ac.uk",
    "https://gitee.com/",
    "https://openai.com/",
    "https://deci.ai/",
    "https://llama.meta.com/llama3",
    "wikipedia.org",
    "https://huggingface.co",
    "https://monai.io/",
]


def get_all_ast_nodes(ast_nodes):
    for node in ast_nodes:
        yield node
        if "children" in node:
            yield from get_all_ast_nodes(node["children"])


def get_all_references_from_md(md_path):
    parse_markdown = mistune.create_markdown(renderer=mistune.AstRenderer())
    ast = parse_markdown(md_path.read_text(encoding="UTF-8"))

    for node in get_all_ast_nodes(ast):
        if node["type"] == "image":
            yield node["src"]
        elif node["type"] == "link":
            yield node["link"]


def validate_colab_url(url: str) -> bool:
    OPENVINO_COLAB_URL_PREFIX = "https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/"

    if not url.startswith(OPENVINO_COLAB_URL_PREFIX):
        return

    notebook_path = url.split(OPENVINO_COLAB_URL_PREFIX)[1]
    absolute_notebook_path = NOTEBOOKS_ROOT / notebook_path

    if not absolute_notebook_path.exists():
        raise ValueError(f"notebook not found for colab url {url!r}")


def collect_md_files(changed_paths=None):
    """Return README markdown files to check.

    If changed_paths is given, return only README .md files from that list
    plus any README .md files inside directories from that list.
    Otherwise return all README .md files in the repo.
    """
    if changed_paths is None:
        return list(NOTEBOOKS_ROOT.glob("**/*README*.md"))

    md_files = []
    seen = set()
    for p in changed_paths:
        path = NOTEBOOKS_ROOT / p
        if path.is_dir():
            for f in path.glob("**/*README*.md"):
                if f not in seen:
                    seen.add(f)
                    md_files.append(f)
        elif path.is_file() and "README" in path.name and path.suffix == ".md":
            if path not in seen:
                seen.add(path)
                md_files.append(path)
        # Also check READMEs in parent dir of changed files
        # (catches broken relative links when files are renamed/deleted)
        elif path.parent.is_dir():
            for f in path.parent.glob("*README*.md"):
                if f not in seen:
                    seen.add(f)
                    md_files.append(f)
    return md_files


def main():
    parser = argparse.ArgumentParser(description="Check links in README files.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Paths (relative to repo root) to check. If omitted, all READMEs are checked.",
    )
    args = parser.parse_args()

    all_passed = True

    def complain(message):
        nonlocal all_passed
        all_passed = False
        print(message, file=sys.stderr)

    changed_paths = args.paths if args.paths else None
    md_files = collect_md_files(changed_paths)

    if not md_files:
        print("No README files to check.")
        sys.exit(0)

    # url -> list of md files that reference it
    remote_urls = defaultdict(list)

    for md_path in md_files:
        for url in get_all_references_from_md(md_path):
            try:
                components = urllib.parse.urlparse(url)
            except ValueError:
                complain(f"{md_path}: invalid URL reference {url!r}")
                continue

            if not components.path:  # self-link
                continue

            if not components.scheme and not components.netloc:
                # check if it is relative path on file from repo
                file_name = md_path.parent / components.path
                if not file_name.exists():
                    complain(f"{md_path}: invalid URL reference {url!r}")
                continue

            try:
                validate_colab_url(url)
            except ValueError as err:
                complain(f"{md_path}: {err}")

            remote_urls[url].append(md_path)

    # Group URLs by domain, then interleave (round-robin) so consecutive
    # requests go to different domains — fills the per-domain delay with
    # useful work instead of sleeping.
    urls_by_domain = defaultdict(list)
    for url in remote_urls:
        domain = urllib.parse.urlparse(url).netloc
        urls_by_domain[domain].append(url)

    interleaved_urls = []
    domain_queues = list(urls_by_domain.values())
    while domain_queues:
        next_round = []
        for queue in domain_queues:
            interleaved_urls.append(queue.pop(0))
            if queue:
                next_round.append(queue)
        domain_queues = next_round

    last_request_time = {}
    PER_DOMAIN_DELAY = 1.0  # seconds between requests to the same domain

    for url in interleaved_urls:
        md_paths = remote_urls[url]
        domain = urllib.parse.urlparse(url).netloc

        # Only sleep if not enough time has passed since last request to this domain
        now = time.monotonic()
        if domain in last_request_time:
            elapsed = now - last_request_time[domain]
            if elapsed < PER_DOMAIN_DELAY:
                time.sleep(PER_DOMAIN_DELAY - elapsed)
        last_request_time[domain] = time.monotonic()

        try:
            status_code = None
            for attempt in range(MAX_RETRIES):
                get = requests.get(url, timeout=10)
                status_code = get.status_code
                if status_code != 429:
                    break
                time.sleep(RETRY_BACKOFF * (2**attempt))

            if status_code not in [200, 202]:
                if status_code == 429:
                    print(f"SKIP - {md_paths[0]}: URL rate-limited after {MAX_RETRIES} retries {url!r}, status code 429")
                    continue
                if status_code in [502, 500, 443, 403, 401] and any(known_url in url for known_url in EXCEPTIONS_URLs):
                    print(f"SKIP - {md_paths[0]}: URL can not be reached {url!r}, status code {status_code}")
                    continue
                for md_path in md_paths:
                    complain(f"{md_path}: URL can not be reached {url!r}, status code {status_code}")
        except Exception as err:
            if any(known_url in url for known_url in EXCEPTIONS_URLs):
                print(f"SKIP - {md_paths[0]}: URL can not be reached {url!r}, error {err}")
            else:
                for md_path in md_paths:
                    complain(f"{md_path}: URL can not be reached {url!r}, error {err}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
