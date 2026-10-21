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


@pytest.fixture
def notebook_bootstrap(tmp_path, monkeypatch):
    notebook = json.loads(BOOTSTRAP_NOTEBOOK.read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "notebook_utils", raising=False)
    importlib.invalidate_caches()

    def retrieve(url, filename):
        Path(filename).write_bytes(COMPATIBLE_UTILS)
        return str(filename), None

    with patch("urllib.request.urlretrieve", side_effect=retrieve) as download:
        yield source, download
    sys.modules.pop("notebook_utils", None)


def test_notebook_bootstrap_refreshes_stale_utils(notebook_bootstrap):
    source, download = notebook_bootstrap
    Path("notebook_utils.py").write_text("OLD_UTILS = True\n", encoding="utf-8")

    exec(compile(source, str(BOOTSTRAP_NOTEBOOK), "exec"), {})

    assert Path("notebook_utils.py").read_bytes() == COMPATIBLE_UTILS
    assert sys.modules["notebook_utils"].SETUP_CALLS == 1
    download.assert_called_once()


def test_notebook_bootstrap_downloads_missing_utils(notebook_bootstrap):
    source, download = notebook_bootstrap

    exec(compile(source, str(BOOTSTRAP_NOTEBOOK), "exec"), {})

    assert sys.modules["notebook_utils"].SETUP_CALLS == 1
    assert Path("notebook_utils.py").read_bytes() == COMPATIBLE_UTILS
    download.assert_called_once()


def test_notebook_bootstrap_keeps_compatible_utils(notebook_bootstrap):
    source, download = notebook_bootstrap
    Path("notebook_utils.py").write_bytes(COMPATIBLE_UTILS)
    exec(compile(source, str(BOOTSTRAP_NOTEBOOK), "exec"), {})
    module = sys.modules["notebook_utils"]

    with patch("importlib.reload", side_effect=AssertionError("compatible module must not be reloaded")):
        exec(compile(source, str(BOOTSTRAP_NOTEBOOK), "exec"), {})

    download.assert_not_called()
    assert sys.modules["notebook_utils"] is module
    assert module.SETUP_CALLS == 2


def test_notebook_bootstrap_refreshes_cached_stale_utils(notebook_bootstrap):
    source, download = notebook_bootstrap
    Path("notebook_utils.py").write_text("OLD_UTILS = True\n", encoding="utf-8")
    module = importlib.import_module("notebook_utils")

    exec(compile(source, str(BOOTSTRAP_NOTEBOOK), "exec"), {})

    assert sys.modules["notebook_utils"] is module
    assert module.SETUP_CALLS == 1
    download.assert_called_once()


def test_notebook_bootstrap_reloads_updated_disk_without_download(notebook_bootstrap):
    source, download = notebook_bootstrap
    utils_path = Path("notebook_utils.py")
    utils_path.write_bytes(b"OLD_UTILS = True\n")
    module = importlib.import_module("notebook_utils")
    utils_path.write_bytes(COMPATIBLE_UTILS)

    exec(compile(source, str(BOOTSTRAP_NOTEBOOK), "exec"), {})

    download.assert_not_called()
    assert sys.modules["notebook_utils"] is module
    assert module.SETUP_CALLS == 1


@pytest.mark.parametrize(
    "error",
    [urllib.error.URLError("offline"), urllib.error.HTTPError("https://example.com", 404, "Not Found", {}, None), TimeoutError("timed out")],
)
def test_notebook_bootstrap_propagates_download_failure(notebook_bootstrap, error):
    source, download = notebook_bootstrap
    original = b"OLD_UTILS = True\n"
    Path("notebook_utils.py").write_bytes(original)
    download.side_effect = error

    with pytest.raises(type(error)):
        exec(compile(source, str(BOOTSTRAP_NOTEBOOK), "exec"), {})

    assert Path("notebook_utils.py").read_bytes() == original
    assert sorted(path.name for path in Path.cwd().iterdir() if path.is_file()) == ["notebook_utils.py"]


def test_notebook_bootstrap_is_consistent_across_notebooks():
    root = BOOTSTRAP_NOTEBOOK.parents[2]
    reference = json.loads(BOOTSTRAP_NOTEBOOK.read_text(encoding="utf-8"))
    reference_cell = next(cell for cell in reference["cells"] if cell["cell_type"] == "code")
    reference_nodes = bootstrap_nodes("".join(reference_cell["source"]))
    for path in (root / "notebooks").glob("*/*.ipynb"):
        if path.name.startswith("test_"):
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell = next((cell for cell in notebook["cells"] if cell["cell_type"] == "code"), None)
        if cell is None:
            continue
        assert bootstrap_nodes("".join(cell["source"])) == reference_nodes, path


def bootstrap_nodes(source):
    nodes = ast.parse(source).body
    start = next(
        i for i, node in enumerate(nodes) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "utils_path" for t in node.targets)
    )
    end = next(i for i, node in enumerate(nodes) if isinstance(node, ast.ImportFrom) and node.module == "notebook_utils")
    return [ast.dump(node) for node in nodes[start : end + 1]]


def test_contributing_bootstrap_matches_notebook():
    root = BOOTSTRAP_NOTEBOOK.parents[2]
    guide = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    example = guide.split("```python\nimport importlib\n", 1)[1].split("```", 1)[0]
    notebook = json.loads(BOOTSTRAP_NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")["source"])
    assert ast.dump(ast.parse("import importlib\n" + example)) == ast.dump(ast.parse(source))


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
