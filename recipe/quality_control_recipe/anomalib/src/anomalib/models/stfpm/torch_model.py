"""PyTorch model for the STFPM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from torch import Tensor, nn

from anomalib.models.components import FeatureExtractor
from anomalib.models.stfpm.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler


class STFPMModel(nn.Module):
    """STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.

    Args:
        layers (list[str]): Layers used for feature extraction
        input_size (tuple[int, int]): Input size for the model.
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
    """

    def __init__(
        self,
        layers: list[str],
        input_size: tuple[int, int],
        backbone: str = "resnet18",
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.teacher_model = FeatureExtractor(backbone=self.backbone, pre_trained=True, layers=layers)
        self.student_model = FeatureExtractor(
            backbone=self.backbone, pre_trained=False, layers=layers, requires_grad=True
        )

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        # Create the anomaly heatmap generator whether tiling is set.
        # TODO: Check whether Tiler is properly initialized here.
        if self.tiler:
            image_size = (self.tiler.tile_size_h, self.tiler.tile_size_w)
        else:
            image_size = input_size
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=image_size)

    def forward(self, images: Tensor) -> Tensor | dict[str, Tensor] | tuple[dict[str, Tensor]]:
        """Forward-pass images into the network.

        During the training mode the model extracts the features from the teacher and student networks.
        During the evaluation mode, it returns the predicted anomaly map.

        Args:
          images (Tensor): Batch of images.

        Returns:
          Teacher and student features when in training mode, otherwise the predicted anomaly maps.
        """
        if self.tiler:
            images = self.tiler.tile(images)
        teacher_features: dict[str, Tensor] = self.teacher_model(images)
        student_features: dict[str, Tensor] = self.student_model(images)
        if self.training:
            output = teacher_features, student_features
        else:
            output = self.anomaly_map_generator(teacher_features=teacher_features, student_features=student_features)
            if self.tiler:
                output = self.tiler.untile(output)

        return output
