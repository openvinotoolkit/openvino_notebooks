#!/usr/bin/env python
# coding: utf-8
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file contains utility functions for use with OpenVINO Notebooks
# https://github.com/openvinotoolkit/openvino_notebooks

from typing import List, NamedTuple, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


class Label(NamedTuple):
    index: int
    color: Tuple
    name: Optional[str] = None


class SegmentationMap(NamedTuple):
    labels: List

    def get_colormap(self):
        return np.array([label.color for label in self.labels])

    def get_labels(self):
        labelnames = [label.name for label in self.labels]
        if any(labelnames):
            return labelnames
        else:
            return None


cityscape_labels = [
    Label(index=0, color=(128, 64, 128), name="road"),
    Label(index=1, color=(244, 35, 232), name="sidewalk"),
    Label(index=2, color=(70, 70, 70), name="building"),
    Label(index=3, color=(102, 102, 156), name="wall"),
    Label(index=4, color=(190, 153, 153), name="fence"),
    Label(index=5, color=(153, 153, 153), name="pole"),
    Label(index=6, color=(250, 170, 30), name="traffic light"),
    Label(index=7, color=(220, 220, 0), name="traffic sign"),
    Label(index=8, color=(107, 142, 35), name="vegetation"),
    Label(index=9, color=(152, 251, 152), name="terrain"),
    Label(index=10, color=(70, 130, 180), name="sky"),
    Label(index=11, color=(220, 20, 60), name="person"),
    Label(index=12, color=(255, 0, 0), name="rider"),
    Label(index=13, color=(0, 0, 142), name="car"),
    Label(index=14, color=(0, 0, 70), name="truck"),
    Label(index=15, color=(0, 60, 100), name="bus"),
    Label(index=16, color=(0, 80, 100), name="train"),
    Label(index=17, color=(0, 0, 230), name="motorcycle"),
    Label(index=18, color=(119, 11, 32), name="bicycle"),
    Label(index=19, color=(0, 0, 0), name="background"),
]

CityScapes = SegmentationMap(cityscape_labels)


def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    if data.max() == data.min():
        raise ValueError(
            "Normalization is not possible because all elements of"
            f"`data` have the same value: {data.max()}."
        )
    return (data - data.min()) / (data.max() - data.min())


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


def to_bgr(image_data) -> np.ndarray:
    """
    Convert image_data from RGB to BGR
    """
    return cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)


def segmentation_map_to_image(result: np.ndarray, colormap: np.ndarray, remove_holes=False):
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    :param result: A single network result in H,W or 1,H,W shape
    :param colormap: a numpy array of shape (num_classes, 3) with an RGB value per class
    :return: an RGB image with int8 images.
    """
    if len(result.shape) != 2 and result.shape[0] != 1:
        raise ValueError(
            f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
        )
    elif result.shape[0] == 1:
        result = result.squeeze(0)

    result = result.astype(np.uint8)

    contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
    mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for label_index, color in enumerate(colormap):
        label_index_map = result == label_index
        label_index_map = label_index_map.astype(np.uint8) * 255
        contours, hierarchies = cv2.findContours(
            label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            mask,
            contours,
            contourIdx=-1,
            color=color.tolist(),
            thickness=cv2.FILLED,
        )
    return mask


def viz_image_and_result(
    source_image: np.ndarray,
    result_image: np.ndarray,
    source_title=None,
    result_title=None,
    labels=None,
    resize=False,
    bgr_to_rgb=False,
    hide_axes=False,
):
    if bgr_to_rgb:
        source_image = to_rgb(source_image)
    if resize:
        result_image = cv2.resize(
            result_image, (source_image.shape[1], source_image.shape[0])
        )
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(source_image)
    ax[0].set_title(source_title)
    ax[1].imshow(result_image)
    if hide_axes:
        for a in ax.ravel():
            a.axis("off")
    if labels:
        colors = labels.get_colormap()
        lines = [
            Line2D(
                [0],
                [0],
                color=[item / 255 for item in c.tolist()],
                linewidth=3,
                linestyle="-",
            )
            for c in colors
        ]
        plt.legend(
            lines,
            labels.get_labels(),
            bbox_to_anchor=(1, 1),
            loc="upper left",
            prop={"size": 12},
        )
    plt.close(fig)
    return fig


def get_cpu_info():
    try:
        import cpuinfo

        cpu = cpuinfo.get_cpu_info()["brand_raw"]
    except Exception:
        # OpenVINO installs cpuinfo, but if a different version is installed
        # the command above may not work
        import platform

        cpu = platform.processor()
    return cpu
