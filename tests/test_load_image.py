from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import requests

from utils.notebook_utils import load_image


def make_response(status=200, content=b""):
    response = MagicMock()
    response.content = content
    if status >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status} Client Error: Not Found for url: https://example.com")
    return response


def test_load_image_raises_on_http_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("requests.get", return_value=make_response(404, b"<html>Not Found</html>")):
        with pytest.raises(requests.exceptions.HTTPError):
            load_image("img.png", "https://example.com/missing.png")
    assert not (tmp_path / "img.png").exists()


def test_load_image_raises_if_content_is_not_an_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("requests.get", return_value=make_response(200, b"<html>not an image</html>")):
        with pytest.raises(ValueError, match="decode"):
            load_image("img.png", "https://example.com/page.html")
    assert not (tmp_path / "img.png").exists()


def test_load_image_requires_url_when_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="url"):
        load_image("img.png")


def test_load_image_downloads_and_saves_valid_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, encoded = cv2.imencode(".png", np.zeros((4, 4, 3), dtype="uint8"))
    with patch("requests.get", return_value=make_response(200, encoded.tobytes())):
        image = load_image("img.png", "https://example.com/ok.png")
    assert image.shape == (4, 4, 3)
    assert (tmp_path / "img.png").exists()
