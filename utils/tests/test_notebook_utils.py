import ast
import gzip
import importlib
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from utils import notebook_utils
from utils.notebook_utils import download_file, download_ir_model, setup_notebook_venv

BOOTSTRAP_NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks/hello-world/hello-world.ipynb"
COMPATIBLE_UTILS = b"SETUP_CALLS = 0\ndef setup_notebook_venv(*args, **kwargs):\n    global SETUP_CALLS\n    SETUP_CALLS += 1\n"
CMD_HELPER = b"CMD_HELPER = True\n"
HELPER_URL = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/"


def bootstrap_source(helper_files):
    notebook = json.loads(BOOTSTRAP_NOTEBOOK.read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    return source.replace('helper_files = ["notebook_utils.py"]', f"helper_files = {helper_files!r}".replace("'", '"'))


def run_bootstrap(source):
    exec(compile(source, str(BOOTSTRAP_NOTEBOOK), "exec"), {})


@pytest.fixture
def notebook_bootstrap(tmp_path, monkeypatch):
    """Notebook folder at `<tmp>/notebooks/demo`, with `urlopen` serving the helpers from `served`."""
    notebook_dir = tmp_path / "notebooks" / "demo"
    notebook_dir.mkdir(parents=True)
    monkeypatch.chdir(notebook_dir)
    monkeypatch.syspath_prepend(str(notebook_dir))
    monkeypatch.delitem(sys.modules, "notebook_utils", raising=False)
    importlib.invalidate_caches()
    served = {"notebook_utils.py": COMPATIBLE_UTILS, "cmd_helper.py": CMD_HELPER}

    def urlopen(url, timeout):
        assert url.startswith(HELPER_URL) and timeout == 10
        response = MagicMock()
        response.__enter__.return_value.read.return_value = served[url[len(HELPER_URL) :]]
        return response

    with patch("urllib.request.urlopen", side_effect=urlopen) as download:
        yield SimpleNamespace(source=bootstrap_source(["notebook_utils.py", "cmd_helper.py"]), download=download, repo_utils=tmp_path / "utils")
    sys.modules.pop("notebook_utils", None)


def test_notebook_bootstrap_downloads_missing_helpers(notebook_bootstrap):
    run_bootstrap(notebook_bootstrap.source)

    assert Path("notebook_utils.py").read_bytes() == COMPATIBLE_UTILS
    assert Path("cmd_helper.py").read_bytes() == CMD_HELPER
    assert sys.modules["notebook_utils"].SETUP_CALLS == 1
    assert [c.args[0] for c in notebook_bootstrap.download.call_args_list] == [HELPER_URL + "notebook_utils.py", HELPER_URL + "cmd_helper.py"]


def test_notebook_bootstrap_overwrites_downloaded_helpers(notebook_bootstrap):
    Path("notebook_utils.py").write_text("OLD_UTILS = True\n", encoding="utf-8")
    Path("cmd_helper.py").write_text("OLD_HELPER = True\n", encoding="utf-8")

    run_bootstrap(notebook_bootstrap.source)

    assert Path("notebook_utils.py").read_bytes() == COMPATIBLE_UTILS
    assert Path("cmd_helper.py").read_bytes() == CMD_HELPER
    assert sys.modules["notebook_utils"].SETUP_CALLS == 1


def test_notebook_bootstrap_copies_helpers_from_repository(notebook_bootstrap):
    repo_utils = notebook_bootstrap.repo_utils
    repo_utils.mkdir()
    (repo_utils / "notebook_utils.py").write_bytes(COMPATIBLE_UTILS)
    (repo_utils / "cmd_helper.py").write_bytes(b"REPO_HELPER = True\n")
    Path("cmd_helper.py").write_text("OLD_HELPER = True\n", encoding="utf-8")

    run_bootstrap(notebook_bootstrap.source)

    notebook_bootstrap.download.assert_not_called()
    assert Path("notebook_utils.py").read_bytes() == COMPATIBLE_UTILS
    assert Path("cmd_helper.py").read_bytes() == b"REPO_HELPER = True\n"
    assert sys.modules["notebook_utils"].SETUP_CALLS == 1


def test_notebook_bootstrap_requires_listed_helpers_in_repository(notebook_bootstrap):
    notebook_bootstrap.repo_utils.mkdir()
    (notebook_bootstrap.repo_utils / "notebook_utils.py").write_bytes(COMPATIBLE_UTILS)

    with pytest.raises(FileNotFoundError):
        run_bootstrap(notebook_bootstrap.source)

    notebook_bootstrap.download.assert_not_called()


DOWNLOAD_ERRORS = [
    urllib.error.URLError("offline"),
    urllib.error.HTTPError("https://example.com", 404, "Not Found", {}, None),
    TimeoutError("timed out"),
]


@pytest.mark.parametrize("error", DOWNLOAD_ERRORS)
@pytest.mark.parametrize("fails_on", ["open", "read"])
def test_notebook_bootstrap_keeps_existing_helpers_when_download_fails(notebook_bootstrap, capsys, error, fails_on):
    Path("notebook_utils.py").write_bytes(COMPATIBLE_UTILS)
    Path("cmd_helper.py").write_bytes(b"OLD_HELPER = True\n")
    if fails_on == "open":
        notebook_bootstrap.download.side_effect = error
    else:
        notebook_bootstrap.download.side_effect = None
        notebook_bootstrap.download.return_value.__enter__.return_value.read.side_effect = error

    run_bootstrap(notebook_bootstrap.source)

    assert Path("notebook_utils.py").read_bytes() == COMPATIBLE_UTILS
    assert Path("cmd_helper.py").read_bytes() == b"OLD_HELPER = True\n"
    assert sys.modules["notebook_utils"].SETUP_CALLS == 1
    assert "Could not update cmd_helper.py, using the existing copy" in capsys.readouterr().out


@pytest.mark.parametrize("error", DOWNLOAD_ERRORS)
def test_notebook_bootstrap_propagates_download_failure_without_copy(notebook_bootstrap, error):
    notebook_bootstrap.download.side_effect = error

    with pytest.raises(type(error)):
        run_bootstrap(notebook_bootstrap.source)

    assert list(Path.cwd().iterdir()) == []


def test_notebook_bootstrap_is_consistent_across_notebooks():
    root = BOOTSTRAP_NOTEBOOK.parents[2]
    reference = json.loads(BOOTSTRAP_NOTEBOOK.read_text(encoding="utf-8"))
    reference_cell = next(cell for cell in reference["cells"] if cell["cell_type"] == "code")
    _, reference_nodes = bootstrap_nodes("".join(reference_cell["source"]))
    for path in (root / "notebooks").glob("*/*.ipynb"):
        if path.name.startswith("test_"):
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell = next((cell for cell in notebook["cells"] if cell["cell_type"] == "code"), None)
        if cell is None:
            continue
        helper_files, nodes = bootstrap_nodes("".join(cell["source"]))
        assert nodes == reference_nodes, path
        assert helper_files[0] == "notebook_utils.py" and len(set(helper_files)) == len(helper_files), path
        assert all((root / "utils" / name).is_file() for name in helper_files), path


def bootstrap_nodes(source):
    """Return the `helper_files` list and the dumped statements that follow it up to the `notebook_utils` import."""
    nodes = ast.parse(source).body
    start = next(
        i for i, node in enumerate(nodes) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "helper_files" for t in node.targets)
    )
    end = next(i for i, node in enumerate(nodes) if isinstance(node, ast.ImportFrom) and node.module == "notebook_utils")
    return ast.literal_eval(nodes[start].value), [ast.dump(node) for node in nodes[start + 1 : end + 1]]


def test_contributing_bootstrap_matches_notebook():
    root = BOOTSTRAP_NOTEBOOK.parents[2]
    guide = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    example = guide.split("```python\nimport urllib.request\n", 1)[1].split("```", 1)[0]
    notebook = json.loads(BOOTSTRAP_NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")["source"])
    assert ast.dump(ast.parse("import urllib.request\n" + example)) == ast.dump(ast.parse(source))


@pytest.mark.parametrize("notebook_name", ["notebook", "notebook.py", "nested/notebook.ipynb"])
def test_setup_notebook_venv_rejects_invalid_notebook_name(notebook_name, tmp_path):
    with pytest.raises(ValueError, match="ending in .ipynb"):
        setup_notebook_venv(notebook_name, venv_root=tmp_path)


def test_load_venv_pth_files_skips_non_install_cells(monkeypatch, tmp_path):
    monkeypatch.setattr(notebook_utils, "_venv_site_dirs", [str(tmp_path)])
    monkeypatch.setattr(notebook_utils, "_processed_pth_files", set())
    result = SimpleNamespace(info=SimpleNamespace(raw_cell="x = 1"))

    with patch("pathlib.Path.glob", side_effect=AssertionError("unexpected .pth scan")):
        notebook_utils._load_venv_pth_files(result)


@pytest.mark.parametrize(
    "cell",
    ["%pip install package", "!pip install package", "!python -m pip install package", "pip_helper.pip_install('package')"],
)
def test_load_venv_pth_files_scans_install_cells(monkeypatch, tmp_path, cell):
    (tmp_path / "editable.pth").write_text("/editable/source\n", encoding="utf-8")
    monkeypatch.setattr(notebook_utils, "_venv_site_dirs", [str(tmp_path)])
    monkeypatch.setattr(notebook_utils, "_parent_site_dirs", [])
    monkeypatch.setattr(notebook_utils, "_processed_pth_files", set())
    result = SimpleNamespace(info=SimpleNamespace(raw_cell=cell))

    with patch("site.addpackage") as addpackage:
        notebook_utils._load_venv_pth_files(result)

    addpackage.assert_called_once_with(str(tmp_path), "editable.pth", None)


def test_download_file_accepts_gzip_response(tmp_path):
    content = b"model data" * 100
    response = MagicMock()
    response.headers = {
        "Content-Encoding": "gzip",
        "Content-Length": str(len(gzip.compress(content))),
    }
    response.iter_content.return_value = iter([content])

    with patch("requests.get", return_value=response), patch("tqdm.notebook.tqdm_notebook"):
        path = download_file("https://example.com/model.bin", directory=tmp_path, show_progress=False)

    assert path.read_bytes() == content
    assert not (tmp_path / "model.bin.part").exists()


@pytest.mark.parametrize(
    ("xml_url", "bin_url"),
    [
        ("https://example.com/model.xml", "https://example.com/model.bin"),
        ("https://example.com/model.xml?token=abc", "https://example.com/model.bin?token=abc"),
        ("https://example.com/model.xml#section", "https://example.com/model.bin#section"),
        ("https://example.com/model.xml?token=abc#section", "https://example.com/model.bin?token=abc#section"),
        ("https://example.com/model.v1.xml?download=1", "https://example.com/model.v1.bin?download=1"),
    ],
)
def test_download_ir_model_preserves_query_and_fragment(xml_url, bin_url, tmp_path):
    xml_path = tmp_path / "model.xml"
    with patch("utils.notebook_utils.download_file", return_value=xml_path) as download:
        result = download_ir_model(xml_url, destination_folder=tmp_path)

    assert result == xml_path
    assert download.call_count == 2
    assert download.call_args_list[0].args == (xml_url,)
    assert download.call_args_list[0].kwargs == {"directory": tmp_path, "show_progress": False}
    assert download.call_args_list[1].args == (bin_url,)
    assert download.call_args_list[1].kwargs == {"directory": tmp_path}
