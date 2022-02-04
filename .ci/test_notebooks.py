import urllib.error
import urllib.request
from pathlib import Path
from typing import Set

import toml
from pip._internal.req import parse_requirements


def get_parsed_requirements(requirements_file: str) -> Set:
    """
    Returns a set of requirements that are defined in `requirements_file`,
    without versions
    """
    requirements_set = set()
    parsed_requirements = parse_requirements(requirements_file, session=False)
    separators = ("=", "<", ">", "[")
    for req in parsed_requirements:
        requirement = req.requirement
        for separator in separators:
            requirement = requirement.replace(separator, "|")

        requirements_set.add(requirement.split("|")[0])
    return requirements_set


def test_readme():
    """
    Test that all notebooks have a README file and exist in the Notebooks README
    """
    notebooks_readme = Path("notebooks/README.md").read_text(encoding="utf-8")
    for item in Path("notebooks").iterdir():
        if item.is_dir():
            # item is a notebook directory
            notebook_dir = item.relative_to("notebooks")
            if str(notebook_dir)[0].isdigit():
                assert "README.md" in [
                    filename.name for filename in item.iterdir()
                ], f"README not found in {item}"
                assert (
                    str(notebook_dir) in notebooks_readme
                ), f"{item} not found in notebooks README"


def test_requirements_docker():
    """
    Test that requirements.txt is a subset of Docker requirements in Pipfile
    This test does not check requirements versions, it only verifies existence
    """
    with open(".docker/Pipfile") as f:
        pipfile_contents = toml.load(f)
        docker_requirements = set(list(pipfile_contents["packages"].keys()))

    pip_requirements = get_parsed_requirements("requirements.txt")
    assert pip_requirements.issubset(
        docker_requirements
    ), f"Docker Pipfile misses: {pip_requirements.difference(docker_requirements)}"


def test_requirements_binder():
    """
    Test that requirements.txt is a subset of Binder requirements
    This test does not check requirements versions, it only verifies existence
    """
    pip_requirements = get_parsed_requirements("requirements.txt")
    binder_requirements = get_parsed_requirements(".binder/requirements.txt")
    assert pip_requirements.issubset(
        binder_requirements
    ), f"Binder requirements misses: {pip_requirements.difference(binder_requirements)}"


def test_urls_exist():
    """
    Test that urls that may be cached still exist on the server
    """
    urls = [
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "https://github.com/onnx/models/raw/master/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx",
        "https://storage.openvinotoolkit.org/data/test_data/openvino_notebooks/kits19/case_00030.zip"
    ]
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    for url in urls:
        try:
            urlobject = urllib.request.urlopen(url, timeout=10)
            assert urlobject.status == 200
        except urllib.error.HTTPError:
            print(f"Downloading {url} failed")
            raise

