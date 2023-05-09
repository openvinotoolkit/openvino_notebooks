"""FastFlow Torch Model Implementation."""

# Original Code
# Copyright (c) 2022 @gathierry
# https://github.com/gathierry/FastFlow/.
# SPDX-License-Identifier: Apache-2.0
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable

import timm
import torch
from FrEIA.framework import SequenceINN
from timm.models.cait import Cait
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn

from anomalib.models.components.flow import AllInOneBlock
from anomalib.models.fastflow.anomaly_map import AnomalyMapGenerator


def subnet_conv_func(kernel_size: int, hidden_ratio: float) -> Callable:
    """Subnet Convolutional Function.

    Callable class or function ``f``, called as ``f(channels_in, channels_out)`` and
        should return a torch.nn.Module.
        Predicts coupling coefficients :math:`s, t`.

    Args:
        kernel_size (int): Kernel Size
        hidden_ratio (float): Hidden ratio to compute number of hidden channels.

    Returns:
        Callable: Sequential for the subnet constructor.
    """

    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def create_fast_flow_block(
    input_dimensions: list[int],
    conv3x3_only: bool,
    hidden_ratio: float,
    flow_steps: int,
    clamp: float = 2.0,
) -> SequenceINN:
    """Create NF Fast Flow Block.

    This is to create Normalizing Flow (NF) Fast Flow model block based on
    Figure 2 and Section 3.3 in the paper.

    Args:
        input_dimensions (list[int]): Input dimensions (Channel, Height, Width)
        conv3x3_only (bool): Boolean whether to use conv3x3 only or conv3x3 and conv1x1.
        hidden_ratio (float): Ratio for the hidden layer channels.
        flow_steps (int): Flow steps.
        clamp (float, optional): Clamp. Defaults to 2.0.

    Returns:
        SequenceINN: FastFlow Block.
    """
    nodes = SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastflowModel(nn.Module):
    """FastFlow.

    Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.

    Args:
        input_size (tuple[int, int]): Model input size.
        backbone (str): Backbone CNN network
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        flow_steps (int, optional): Flow steps.
        conv3x3_only (bool, optinoal): Use only conv3x3 in fast_flow model. Defaults to False.
        hidden_ratio (float, optional): Ratio to calculate hidden var channels. Defaults to 1.0.

    Raises:
        ValueError: When the backbone is not supported.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str,
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        self.input_size = input_size

        if backbone in ("cait_m48_448", "deit_base_distilled_patch16_384"):
            self.feature_extractor = timm.create_model(backbone, pretrained=pre_trained)
            channels = [768]
            scales = [16]
        elif backbone in ("resnet18", "wide_resnet50_2"):
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for channel, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                        elementwise_affine=True,
                    )
                )
        else:
            raise ValueError(
                f"Backbone {backbone} is not supported. List of available backbones are "
                "[cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2]."
            )

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.fast_flow_blocks = nn.ModuleList()
        for channel, scale in zip(channels, scales):
            self.fast_flow_blocks.append(
                create_fast_flow_block(
                    input_dimensions=[channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

    def forward(self, input_tensor: Tensor) -> Tensor | list[Tensor] | tuple[list[Tensor]]:
        """Forward-Pass the input to the FastFlow Model.

        Args:
            input_tensor (Tensor): Input tensor.

        Returns:
            Tensor | list[Tensor] | tuple[list[Tensor]]: During training, return
                (hidden_variables, log-of-the-jacobian-determinants).
                During the validation/test, return the anomaly map.
        """

        return_val: Tensor | list[Tensor] | tuple[list[Tensor]]

        self.feature_extractor.eval()
        if isinstance(self.feature_extractor, VisionTransformer):
            features = self._get_vit_features(input_tensor)
        elif isinstance(self.feature_extractor, Cait):
            features = self._get_cait_features(input_tensor)
        else:
            features = self._get_cnn_features(input_tensor)

        # Compute the hidden variable f: X -> Z and log-likelihood of the jacobian
        # (See Section 3.3 in the paper.)
        # NOTE: output variable has z, and jacobian tuple for each fast-flow blocks.
        hidden_variables: list[Tensor] = []
        log_jacobians: list[Tensor] = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)

        return_val = (hidden_variables, log_jacobians)

        if not self.training:
            return_val = self.anomaly_map_generator(hidden_variables)

        return return_val

    def _get_cnn_features(self, input_tensor: Tensor) -> list[Tensor]:
        """Get CNN-based features.

        Args:
            input_tensor (Tensor): Input Tensor.

        Returns:
            list[Tensor]: List of features.
        """
        features = self.feature_extractor(input_tensor)
        features = [self.norms[i](feature) for i, feature in enumerate(features)]
        return features

    def _get_cait_features(self, input_tensor: Tensor) -> list[Tensor]:
        """Get Class-Attention-Image-Transformers (CaiT) features.

        Args:
            input_tensor (Tensor): Input Tensor.

        Returns:
            list[Tensor]: List of features.
        """
        feature = self.feature_extractor.patch_embed(input_tensor)
        feature = feature + self.feature_extractor.pos_embed
        feature = self.feature_extractor.pos_drop(feature)
        for i in range(41):  # paper Table 6. Block Index = 40
            feature = self.feature_extractor.blocks[i](feature)
        batch_size, _, num_channels = feature.shape
        feature = self.feature_extractor.norm(feature)
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        features = [feature]
        return features

    def _get_vit_features(self, input_tensor: Tensor) -> list[Tensor]:
        """Get Vision Transformers (ViT) features.

        Args:
            input_tensor (Tensor): Input Tensor.

        Returns:
            list[Tensor]: List of features.
        """
        feature = self.feature_extractor.patch_embed(input_tensor)
        cls_token = self.feature_extractor.cls_token.expand(feature.shape[0], -1, -1)
        if self.feature_extractor.dist_token is None:
            feature = torch.cat((cls_token, feature), dim=1)
        else:
            feature = torch.cat(
                (
                    cls_token,
                    self.feature_extractor.dist_token.expand(feature.shape[0], -1, -1),
                    feature,
                ),
                dim=1,
            )
        feature = self.feature_extractor.pos_drop(feature + self.feature_extractor.pos_embed)
        for i in range(8):  # paper Table 6. Block Index = 7
            feature = self.feature_extractor.blocks[i](feature)
        feature = self.feature_extractor.norm(feature)
        feature = feature[:, 2:, :]
        batch_size, _, num_channels = feature.shape
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        features = [feature]
        return features
