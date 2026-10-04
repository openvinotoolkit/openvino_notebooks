import gzip
from unittest.mock import MagicMock, patch

from utils.notebook_utils import download_file


def test_download_file_accepts_gzip_response(tmp_path):
    content = b"model data" * 100
    response = MagicMock()
    response.headers = {
        "Content-Encoding": "gzip",
        "Content-Length": str(len(gzip.compress(content))),
    }
    response.iter_content.return_value = iter([content])

    with patch("requests.get", return_value=response), patch("tqdm.notebook.tqdm_notebook"):
        path = download_file(
            "https://example.com/model.bin", directory=tmp_path, show_progress=False
        )

    assert path.read_bytes() == content
    assert not (tmp_path / "model.bin.part").exists()
