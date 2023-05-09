"""Image Utils."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
import torch.nn.functional as F
from torch import Tensor
from torchvision.datasets.folder import IMG_EXTENSIONS


def get_image_filenames(path: str | Path) -> list[Path]:
    """Get image filenames.

    Args:
        path (str | Path): Path to image or image-folder.

    Returns:
        list[Path]: List of image filenames

    """
    image_filenames: list[Path]

    if isinstance(path, str):
        path = Path(path)

    if path.is_file() and path.suffix in IMG_EXTENSIONS:
        image_filenames = [path]

    if path.is_dir():
        image_filenames = [p for p in path.glob("**/*") if p.suffix in IMG_EXTENSIONS]

    if not image_filenames:
        raise ValueError(f"Found 0 images in {path}")

    return image_filenames


def duplicate_filename(path: str | Path) -> Path:
    """Check and duplicate filename.

    This function checks the path and adds a suffix if it already exists on the file system.

    Args:
        path (str | Path): Input Path

    Examples:
        >>> path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> path.exists()
        True

        If we pass this to ``duplicate_filename`` function we would get the following:
        >>> duplicate_filename(path)
        PosixPath('datasets/MVTec/bottle/test/broken_large/000_1.png')

    Returns:
        Path: Duplicated output path.
    """

    if isinstance(path, str):
        path = Path(path)

    i = 0
    while True:
        duplicated_path = path if i == 0 else path.parent / (path.stem + f"_{i}" + path.suffix)
        if not duplicated_path.exists():
            break
        i += 1

    return duplicated_path


def generate_output_image_filename(input_path: str | Path, output_path: str | Path) -> Path:
    """Generate an output filename to save the inference image.

    This function generates an output filaname by checking the input and output filenames. Input path is
    the input to infer, and output path is the path to save the output predictions specified by the user.

    The function expects ``input_path`` to always be a file, not a directory. ``output_path`` could be a
    filename or directory. If it is a filename, the function checks if the specified filename exists on
    the file system. If yes, the function calls ``duplicate_filename`` to duplicate the filename to avoid
    overwriting the existing file. If ``output_path`` is a directory, this function adds the parent and
    filenames of ``input_path`` to ``output_path``.

    Args:
        input_path (str | Path): Path to the input image to infer.
        output_path (str | Path): Path to output to save the predictions.
            Could be a filename or a directory.

    Examples:
        >>> input_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> output_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> generate_output_image_filename(input_path, output_path)
        PosixPath('datasets/MVTec/bottle/test/broken_large/000_1.png')

        >>> input_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> output_path = Path("results/images")
        >>> generate_output_image_filename(input_path, output_path)
        PosixPath('results/images/broken_large/000.png')

    Raises:
        ValueError: When the ``input_path`` is not a file.

    Returns:
        Path: The output filename to save the output predictions from the inferencer.
    """

    if isinstance(input_path, str):
        input_path = Path(input_path)

    if isinstance(output_path, str):
        output_path = Path(output_path)

    # This function expects an ``input_path`` that is a file. This is to check if output_path
    if input_path.is_file() is False:
        raise ValueError("input_path is expected to be a file to generate a proper output filename.")

    file_path: Path
    if output_path.is_dir():
        # If the output is a directory, then add parent directory name
        # and filename to the path. This is to ensure we do not overwrite
        # images and organize based on the categories.
        file_path = output_path / input_path.parent.name / input_path.name
    else:
        file_path = output_path

    # This new ``file_path`` might contain a directory path yet to be created.
    # Create the parent directory to avoid such cases.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.is_file():
        warnings.warn(f"{output_path} already exists. Renaming the file to avoid overwriting.")
        file_path = duplicate_filename(file_path)

    return file_path


def get_image_height_and_width(image_size: int | tuple[int, int]) -> tuple[int, int]:
    """Get image height and width from ``image_size`` variable.

    Args:
        image_size (int | tuple[int, int] | None, optional): Input image size.

    Raises:
        ValueError: Image size not None, int or tuple.

    Examples:
        >>> get_image_height_and_width(image_size=256)
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256))
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256, 3))
        (256, 256)

        >>> get_image_height_and_width(image_size=256.)
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_height_and_width
        ValueError: ``image_size`` could be either int or tuple[int, int]

    Returns:
        tuple[int | None, int | None]: A tuple containing image height and width values.
    """
    if isinstance(image_size, int):
        height_and_width = (image_size, image_size)
    elif isinstance(image_size, tuple):
        height_and_width = int(image_size[0]), int(image_size[1])
    else:
        raise ValueError("``image_size`` could be either int or tuple[int, int]")

    return height_and_width


def read_image(path: str | Path, image_size: int | tuple[int, int] | None = None) -> np.ndarray:
    """Read image from disk in RGB format.

    Args:
        path (str, Path): path to the image file

    Example:
        >>> image = read_image("test_image.jpg")

    Returns:
        image as numpy array
    """
    path = path if isinstance(path, str) else str(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_size:
        # This part is optional, where the user wants to quickly resize the image
        # with a one-liner code. This would particularly be useful especially when
        # prototyping new ideas.
        height, width = get_image_height_and_width(image_size)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)

    return image


def read_depth_image(path: str | Path) -> np.ndarray:
    """Read tiff depth image from disk.

    Args:
        path (str, Path): path to the image file

    Example:
        >>> image = read_depth_image("test_image.tiff")

    Returns:
        image as numpy array
    """
    path = path if isinstance(path, str) else str(path)
    image = tiff.imread(path)

    return image


def pad_nextpow2(batch: Tensor) -> Tensor:
    """Compute required padding from input size and return padded images.

    Finds the largest dimension and computes a square image of dimensions that are of the power of 2.
    In case the image dimension is odd, it returns the image with an extra padding on one side.

    Args:
        batch (Tensor): Input images

    Returns:
        batch: Padded batch
    """
    # find the largest dimension
    l_dim = 2 ** math.ceil(math.log(max(*batch.shape[-2:]), 2))
    padding_w = [math.ceil((l_dim - batch.shape[-2]) / 2), math.floor((l_dim - batch.shape[-2]) / 2)]
    padding_h = [math.ceil((l_dim - batch.shape[-1]) / 2), math.floor((l_dim - batch.shape[-1]) / 2)]
    padded_batch = F.pad(batch, pad=[*padding_h, *padding_w])
    return padded_batch
