"""PyTorch model for DFM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anomalib.models.components import PCA, DynamicBufferModule, FeatureExtractor


class SingleClassGaussian(DynamicBufferModule):
    """Model Gaussian distribution over a set of points."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean_vec", Tensor())
        self.register_buffer("u_mat", Tensor())
        self.register_buffer("sigma_mat", Tensor())

        self.mean_vec: Tensor
        self.u_mat: Tensor
        self.sigma_mat: Tensor

    def fit(self, dataset: Tensor) -> None:
        """Fit a Gaussian model to dataset X.

        Covariance matrix is not calculated directly using:
        ``C = X.X^T``
        Instead, it is represented in terms of the Singular Value Decomposition of X:
        ``X = U.S.V^T``
        Hence,
        ``C = U.S^2.U^T``
        This simplifies the calculation of the log-likelihood without requiring full matrix inversion.

        Args:
            dataset (Tensor): Input dataset to fit the model.
        """

        num_samples = dataset.shape[1]
        self.mean_vec = torch.mean(dataset, dim=1)
        data_centered = (dataset - self.mean_vec.reshape(-1, 1)) / math.sqrt(num_samples)
        self.u_mat, self.sigma_mat, _ = torch.linalg.svd(data_centered, full_matrices=False)

    def score_samples(self, features: Tensor) -> Tensor:
        """Compute the NLL (negative log likelihood) scores.

        Args:
            features (Tensor): semantic features on which density modeling is performed.

        Returns:
            nll (Tensor): Torch tensor of scores
        """
        features_transformed = torch.matmul(features - self.mean_vec, self.u_mat / self.sigma_mat)
        nll = torch.sum(features_transformed * features_transformed, dim=1) + 2 * torch.sum(torch.log(self.sigma_mat))
        return nll

    def forward(self, dataset: Tensor) -> None:
        """Provides the same functionality as `fit`.

        Transforms the input dataset based on singular values calculated earlier.

        Args:
            dataset (Tensor): Input dataset
        """
        self.fit(dataset)


class DFMModel(nn.Module):
    """Model for the DFM algorithm.

    Args:
        backbone (str): Pre-trained model backbone.
        layer (str): Layer from which to extract features.
        input_size (tuple[int, int]): Input size for the model.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        pooling_kernel_size (int, optional): Kernel size to pool features extracted from the CNN.
        n_comps (float, optional): Ratio from which number of components for PCA are calculated. Defaults to 0.97.
        score_type (str, optional): Scoring type. Options are `fre` and `nll`. Defaults to "fre". Anomaly
        segmentation is supported with `fre` only. If using `nll`, set `task` in config.yaml to classification
    """

    def __init__(
        self,
        backbone: str,
        layer: str,
        input_size: tuple[int, int],
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
        n_comps: float = 0.97,
        score_type: str = "fre",
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.n_components = n_comps
        self.pca_model = PCA(n_components=self.n_components)
        self.gaussian_model = SingleClassGaussian()
        self.score_type = score_type
        self.layer = layer
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)
        self.feature_extractor = FeatureExtractor(
            backbone=self.backbone, pre_trained=pre_trained, layers=[layer]
        ).eval()

    def fit(self, dataset: Tensor) -> None:
        """Fit a pca transformation and a Gaussian model to dataset.

        Args:
            dataset (Tensor): Input dataset to fit the model.
        """

        self.pca_model.fit(dataset)
        if self.score_type == "nll":
            features_reduced = self.pca_model.transform(dataset)
            self.gaussian_model.fit(features_reduced.T)

    def score(self, features: Tensor, feature_shapes: tuple) -> Tensor:
        """Compute scores.

        Scores are either PCA-based feature reconstruction error (FRE) scores or
        the Gaussian density-based NLL scores

        Args:
            features (torch.Tensor): semantic features on which PCA and density modeling is performed.
            feature_shapes  (tuple): shape of `features` tensor. Used to generate anomaly map of correct shape.

        Returns:
            score (Tensor): numpy array of scores
        """
        feats_projected = self.pca_model.transform(features)
        if self.score_type == "nll":
            score = self.gaussian_model.score_samples(feats_projected)
        elif self.score_type == "fre":
            feats_reconstructed = self.pca_model.inverse_transform(feats_projected)
            fre = torch.square(features - feats_reconstructed).reshape(feature_shapes)
            fre_map = torch.unsqueeze(torch.sum(fre, dim=1), 1)
            score_map = F.interpolate(fre_map, size=self.input_size, mode="bilinear", align_corners=False)
            score = torch.sum(torch.square(features - feats_reconstructed), dim=1)
        else:
            raise ValueError(f"unsupported score type: {self.score_type}")

        if self.score_type == "nll":
            output = score
        else:
            output = score_map, score

        return output

    def get_features(self, batch: Tensor) -> Tensor:
        """Extract features from the pretrained network.

        Args:
            batch (Tensor): Image batch.

        Returns:
            Tensor: Tensor containing extracted features.
        """
        self.feature_extractor.eval()
        features = self.feature_extractor(batch)[self.layer]
        batch_size = len(features)
        if self.pooling_kernel_size > 1:
            features = F.avg_pool2d(input=features, kernel_size=self.pooling_kernel_size)
        feature_shapes = features.shape
        features = features.view(batch_size, -1).detach()
        if self.training:
            output = features
        else:
            output = (features, feature_shapes)

        return output

    def forward(self, batch: Tensor) -> Tensor:
        """Computer score from input images.

        Args:
            batch (Tensor): Input images

        Returns:
            Tensor: Scores
        """
        feature_vector, feature_shapes = self.get_features(batch)
        return self.score(feature_vector.view(feature_vector.shape[:2]), feature_shapes)
