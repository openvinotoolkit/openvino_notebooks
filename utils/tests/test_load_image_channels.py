from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from utils.notebook_utils import load_image


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_load_image_returns_same_bgr_pixels_before_and_after_caching(tmp_path, channels):
    shape = (3, 4) if channels == 1 else (3, 4, channels)
    original = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    ok, encoded = cv2.imencode(".png", original)
    assert ok
    response = Mock(content=encoded.tobytes())
    destination = tmp_path / "image.png"

    with patch("requests.get", return_value=response) as get:
        downloaded = load_image(str(destination), url="https://example.com/image.png")
        cached = load_image(str(destination), url="https://example.com/image.png")

    assert downloaded.shape == cached.shape == (3, 4, 3)
    np.testing.assert_array_equal(downloaded, cached)
    np.testing.assert_array_equal(downloaded, cv2.imdecode(encoded, cv2.IMREAD_COLOR))
    get.assert_called_once()
    response.raise_for_status.assert_called_once()
