import gzip
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.notebook_utils import download_file, download_ir_model, segmentation_map_to_image


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


@pytest.mark.parametrize(
    "result",
    [
        np.array([[0, 2]], dtype=np.uint8),
        np.array([[0, 1, 2]], dtype=np.uint8),
    ],
)
def test_segmentation_map_rejects_labels_outside_colormap(result):
    colormap = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

    with pytest.raises(ValueError, match="2 classes"):
        segmentation_map_to_image(result, colormap)


def test_segmentation_map_colors_valid_labels():
    colormap = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    result = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    mask = segmentation_map_to_image(result, colormap)

    np.testing.assert_array_equal(mask[0, 0], colormap[0])
    np.testing.assert_array_equal(mask[0, 1], colormap[1])
