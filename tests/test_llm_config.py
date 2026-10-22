import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from utils.llm_config import convert_and_compress_model, convert_and_compress_vlm

MODEL_CONFIG = {"model_id": "org/some-model"}
CONVERTERS = [
    pytest.param(convert_and_compress_model, "openvino_model", id="llm"),
    pytest.param(convert_and_compress_vlm, "openvino_language_model", id="vlm"),
]


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hub = MagicMock()
    hub.HfApi.return_value.repo_exists.return_value = True
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "IPython", MagicMock())
    monkeypatch.setitem(sys.modules, "IPython.display", MagicMock())
    return hub


def create_files(model_dir, names):
    model_dir.mkdir(parents=True)
    for name in names:
        (model_dir / name).write_text("x")


@pytest.mark.parametrize("convert, weights_name", CONVERTERS)
def test_complete_model_is_reused(fake_hub, convert, weights_name):
    model_dir = Path("some-model") / "INT4_compressed_weights"
    create_files(model_dir, [f"{weights_name}.xml", f"{weights_name}.bin", "openvino_tokenizer.xml", "openvino_detokenizer.xml"])
    assert convert("some-model", MODEL_CONFIG, "INT4") == model_dir
    fake_hub.snapshot_download.assert_not_called()


@pytest.mark.parametrize("convert, weights_name", CONVERTERS)
def test_model_without_weights_file_is_downloaded_again(fake_hub, convert, weights_name):
    model_dir = Path("some-model") / "INT4_compressed_weights"
    create_files(model_dir, [f"{weights_name}.xml", "openvino_tokenizer.xml", "openvino_detokenizer.xml"])
    assert convert("some-model", MODEL_CONFIG, "INT4") == model_dir
    fake_hub.snapshot_download.assert_called_once_with("OpenVINO/some-model-int4-ov", local_dir=model_dir)
