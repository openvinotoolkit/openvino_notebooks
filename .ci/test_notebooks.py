import json
import requests
from pathlib import Path
import pytest
from scarf_pixel import check_scarf_tag
from check_notebooks import NOTEBOOKS_ROOT, unencoded_py_write_issues

import toml
from pip._internal.req import parse_requirements


def get_parsed_requirements(requirements_file: str) -> set:
    """
    Returns a set of requirements that are defined in `requirements_file`,
    without versions
    """
    requirements_set = set()
    ignore_list = [
        "paddlenlp",
        "paddle2onnx",
        "paddlepaddle",
    ]  # temporary ignore paddle
    parsed_requirements = parse_requirements(requirements_file, session=False)
    separators = ("=", "<", ">", "[")
    for req in parsed_requirements:
        requirement = req.requirement
        # requirements for Windows or macOS only
        if ";" in requirement and "linux" not in requirement:
            continue
        if requirement.startswith("git+"):
            requirement = requirement.split("#egg=")[-1]
        for separator in separators:
            requirement = requirement.replace(separator, "|")
        reqname = requirement.split("|")[0]
        if reqname not in ignore_list:
            requirements_set.add(reqname)

    return requirements_set


def test_readme():
    """
    Test that all notebooks have a README file and README file includes Scarg Pixel tag
    """
    for item in Path("notebooks").iterdir():
        if item.is_dir():
            assert "README.md" in [filename.name for filename in item.iterdir()], f"README not found in {item}"
            readme_path = item / "README.md"
            assert check_scarf_tag(
                readme_path
            ), f"README file at path '{str(readme_path)}' does not contain Scarf Pixel tag.\nYou can generate Scarf Pixel tag with the following command:\n    python .ci/scarf_pixel.py -s <PATH>"


def test_requirements_docker():
    """
    Test that requirements.txt is a subset of Docker requirements in Pipfile
    This test does not check requirements versions, it only verifies existence
    """
    with open(".docker/Pipfile") as f:
        pipfile_contents = toml.load(f)
        docker_requirements = set(list(pipfile_contents["packages"].keys()))

    pip_requirements = get_parsed_requirements("requirements.txt")
    assert pip_requirements.issubset(docker_requirements), f"Docker Pipfile misses: {pip_requirements.difference(docker_requirements)}"


def test_requirements_binder():
    """
    Test that requirements.txt is a subset of Binder requirements
    This test does not check requirements versions, it only verifies existence
    """
    pip_requirements = get_parsed_requirements("requirements.txt")
    binder_requirements = get_parsed_requirements(".binder/requirements.txt")
    assert pip_requirements.issubset(binder_requirements), f"Binder requirements misses: {pip_requirements.difference(binder_requirements)}"


@pytest.mark.skip(reason="URL existence is tested in docker_treon")
def test_urls_exist():
    """
    Test that urls that may be cached still exist on the server
    """
    urls = [
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx",
        "https://storage.openvinotoolkit.org/data/test_data/openvino_notebooks/kits19/case_00030.zip",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        response = requests.get(url=url, headers=headers, timeout=30)
        if not response.status_code == 200:
            print(f"Downloading {url} failed")
            raise


def _notebook(*code_cells):
    return {"cells": [{"cell_type": "markdown", "source": ["# Title"]}] + [{"cell_type": "code", "source": [cell]} for cell in code_cells]}


@pytest.mark.parametrize(
    "code",
    [
        'open("gradio_helper.py", "w").write(r.text)',
        "open('gradio_helper.py', 'w').write(r.text)",
        'with open("ov_helper.py", "w") as f:\n    f.write(r.text)',
        'open(f"{name}_helper.py", "w").write(r.text)',
    ],
)
def test_unencoded_py_write_is_reported(code):
    issues = unencoded_py_write_issues(_notebook("import requests", code))
    assert len(issues) == 1
    assert "code cell 2" in issues[0]


@pytest.mark.parametrize(
    "code",
    [
        'open("gradio_helper.py", "w", encoding="utf-8").write(r.text)',
        'open("gradio_helper.py", "wb").write(r.content)',
        'open("gradio_helper.py").read()',
        'open("result.txt", "w").write("ok")',
    ],
)
def test_encoded_or_unrelated_write_is_not_reported(code):
    assert unencoded_py_write_issues(_notebook(code)) == []


def test_notebooks_write_py_files_with_explicit_encoding():
    """Downloaded helpers must be written as UTF-8, otherwise non-ASCII content fails on Windows (cp1252)."""
    failures = []
    for nb_path in sorted(NOTEBOOKS_ROOT.glob("notebooks/**/*.ipynb")):
        if ".ipynb_checkpoints" in nb_path.parts or ".venv" in nb_path.parts:
            continue
        notebook_json = json.loads(nb_path.read_text(encoding="utf-8"))
        failures += [f"{nb_path.relative_to(NOTEBOOKS_ROOT)}: {issue}" for issue in unencoded_py_write_issues(notebook_json)]
    assert not failures, "\n".join(failures)
