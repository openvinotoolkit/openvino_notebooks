import numpy as np
import pytest

from utils.notebook_utils import (
    BinarySegmentation,
    Label,
    segmentation_map_to_image,
    segmentation_map_to_overlay,
    viz_result_image,
)


@pytest.fixture
def sample_colormap():
    return np.array(
        [
            [0, 0, 0],       # class 0: background (black)
            [255, 0, 0],     # class 1: red
            [0, 255, 0],     # class 2: green
        ],
        dtype=np.uint8,
    )


def test_segmentation_map_to_image_2d(sample_colormap):
    result = np.array(
        [
            [0, 1],
            [2, 0],
        ],
        dtype=np.int32,
    )
    mask = segmentation_map_to_image(result, sample_colormap)
    assert mask.shape == (2, 2, 3)
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(mask[0, 1], [255, 0, 0])
    np.testing.assert_array_equal(mask[1, 0], [0, 255, 0])
    np.testing.assert_array_equal(mask[1, 1], [0, 0, 0])


def test_segmentation_map_to_image_3d_leading_one(sample_colormap):
    result = np.array(
        [
            [
                [0, 1],
                [2, 0],
            ]
        ],
        dtype=np.int32,
    )
    mask = segmentation_map_to_image(result, sample_colormap)
    assert mask.shape == (2, 2, 3)
    np.testing.assert_array_equal(mask[0, 1], [255, 0, 0])


def test_segmentation_map_to_image_single_row_2d_not_squeezed(sample_colormap):
    # (1, W) 2D input should not be squeezed to 1D which previously raised IndexError
    result = np.array([[0, 1, 2, 1]], dtype=np.int32)
    mask = segmentation_map_to_image(result, sample_colormap)
    assert mask.shape == (1, 4, 3)
    np.testing.assert_array_equal(mask[0, 1], [255, 0, 0])
    np.testing.assert_array_equal(mask[0, 2], [0, 255, 0])


@pytest.mark.parametrize("shape", [(10,), (1, 1, 4, 4), (2, 4, 4)])
def test_segmentation_map_to_image_invalid_shapes(shape, sample_colormap):
    result = np.zeros(shape, dtype=np.int32)
    with pytest.raises(ValueError, match="Expected result with shape"):
        segmentation_map_to_image(result, sample_colormap)


def test_segmentation_map_to_image_exceeds_class_count(sample_colormap):
    # Colormap has 3 classes, result has 4 unique values
    result = np.array([[0, 1], [2, 3]], dtype=np.int32)
    with pytest.raises(ValueError) as excinfo:
        segmentation_map_to_image(result, sample_colormap)
    # The error message should report class count 3, not raw color array
    assert "Expected max 3 classes in result" in str(excinfo.value)


def test_segmentation_map_to_image_out_of_range_class_indices(sample_colormap):
    # Result has only 2 unique values (<= 3), but index 5 is out of range [0, 2]
    result = np.array([[0, 5], [0, 5]], dtype=np.int32)
    with pytest.raises(ValueError, match="outside the colormap range"):
        segmentation_map_to_image(result, sample_colormap)


def test_segmentation_map_to_image_remove_holes(sample_colormap):
    # Ring of class 1 around a hole of class 0
    result = np.ones((7, 7), dtype=np.int32)
    result[2:5, 2:5] = 0

    mask_with_holes = segmentation_map_to_image(result, sample_colormap, remove_holes=False)
    mask_without_holes = segmentation_map_to_image(result, sample_colormap, remove_holes=True)

    # In remove_holes=True with RETR_EXTERNAL contour mode, the outer contour fills the interior
    assert mask_with_holes.shape == (7, 7, 3)
    assert mask_without_holes.shape == (7, 7, 3)


def test_segmentation_map_to_overlay_2d_grayscale(sample_colormap):
    image = np.full((10, 10), 100, dtype=np.uint8)
    result = np.zeros((10, 10), dtype=np.int32)
    overlay = segmentation_map_to_overlay(image, result, alpha=0.5, colormap=sample_colormap)
    assert overlay.shape == (10, 10, 3)
    assert overlay.dtype == np.uint8


def test_segmentation_map_to_overlay_3d_single_channel(sample_colormap):
    image = np.full((10, 10, 1), 100, dtype=np.uint8)
    result = np.zeros((10, 10), dtype=np.int32)
    overlay = segmentation_map_to_overlay(image, result, alpha=0.5, colormap=sample_colormap)
    assert overlay.shape == (10, 10, 3)
    assert overlay.dtype == np.uint8


def test_segmentation_map_to_overlay_4channel_rgba(sample_colormap):
    image = np.full((10, 10, 4), 100, dtype=np.uint8)
    result = np.zeros((10, 10), dtype=np.int32)
    overlay = segmentation_map_to_overlay(image, result, alpha=0.5, colormap=sample_colormap)
    assert overlay.shape == (10, 10, 3)
    assert overlay.dtype == np.uint8


@pytest.mark.parametrize("alpha", [-0.1, 1.2])
def test_segmentation_map_to_overlay_invalid_alpha(alpha, sample_colormap):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = np.zeros((10, 10), dtype=np.int32)
    with pytest.raises(ValueError, match="Expected alpha to be between 0.0 and 1.0"):
        segmentation_map_to_overlay(image, result, alpha=alpha, colormap=sample_colormap)


def test_viz_result_image_with_segmentation_map():
    pytest.importorskip("matplotlib")
    res = np.zeros((10, 10, 3), dtype=np.uint8)
    src = np.zeros((10, 10, 3), dtype=np.uint8)
    fig = viz_result_image(res, source_image=src, labels=BinarySegmentation)
    assert fig is not None


def test_viz_result_image_with_mocked_matplotlib(monkeypatch):
    import sys
    from unittest.mock import MagicMock

    mock_fig = MagicMock()
    mock_axis = MagicMock()
    mock_ax = np.empty((1, 1), dtype=object)
    mock_ax[0, 0] = mock_axis

    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    mock_line2d = MagicMock()
    mock_lines = MagicMock()
    mock_lines.Line2D = mock_line2d

    mock_matplotlib = MagicMock()
    mock_matplotlib.pyplot = mock_plt
    mock_matplotlib.lines = mock_lines

    monkeypatch.setitem(sys.modules, "matplotlib", mock_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mock_plt)
    monkeypatch.setitem(sys.modules, "matplotlib.lines", mock_lines)

    res = np.zeros((10, 10, 3), dtype=np.uint8)
    labels_list = [
        Label(index=0, color=(0, 0, 0), name="bg"),
        Label(index=1, color=(255, 0, 0), name="fg"),
    ]
    fig = viz_result_image(res, labels=labels_list)
    assert fig == mock_fig
    assert mock_line2d.call_count == 2
    mock_axis.legend.assert_called_once()


def test_viz_result_image_resize_without_source():
    pytest.importorskip("matplotlib")
    res = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="`resize` requires `source_image`"):
        viz_result_image(res, resize=True)
