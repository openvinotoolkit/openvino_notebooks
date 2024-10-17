import matplotlib.pyplot as plt

from typing import Tuple

import torch


from copy import deepcopy
from typing import Tuple
from torchvision.transforms.functional import resize, to_pil_image

import numpy as np

np.random.seed(3)


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, mask in enumerate(masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        plt.axis("off")
        plt.show()


class ResizeLongestSide:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming numpy arrays.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def preprocess_image(image: np.ndarray, resizer):
    resized_image = resizer.apply_image(image)
    resized_image = (resized_image.astype(np.float32) - [123.675, 116.28, 103.53]) / [
        58.395,
        57.12,
        57.375,
    ]
    resized_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)).astype(np.float32), 0)

    # Pad
    h, w = resized_image.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = np.pad(resized_image, ((0, 0), (0, 0), (0, padh), (0, padw)))
    return x


def postprocess_masks(masks: np.ndarray, orig_size, resizer):
    size_before_pad = resizer.get_preprocess_shape(orig_size[0], orig_size[1], masks.shape[-1])
    masks = masks[..., : int(size_before_pad[0]), : int(size_before_pad[1])]
    masks = torch.nn.functional.interpolate(torch.from_numpy(masks), size=orig_size, mode="bilinear", align_corners=False).numpy()
    return masks


class SamImageEncoderModel(torch.nn.Module):
    def __init__(self, predictor) -> None:
        super().__init__()
        self.image_encoder = predictor.model.image_encoder
        self.base_model = predictor.model
        self._bb_feat_sizes = predictor._bb_feat_sizes

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,
    ):
        backbone_out = self.base_model.forward_image(image)

        _, vision_feats, _, _ = self.base_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.base_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.base_model.no_mem_embed

        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])][::-1]

        return feats[-1], feats[:-1]


class SamImageMaskPredictionModel(torch.nn.Module):
    def __init__(
        self, model, multimask_output: bool, use_stability_score: bool = False, return_extra_metrics: bool = False, use_high_res_features: bool = True
    ) -> None:
        super().__init__()
        self.mask_decoder = model.sam_mask_decoder
        self.mask_decoder.use_high_res_features = use_high_res_features
        self.model = model
        self.model.use_high_res_features_in_sam = use_high_res_features
        self.img_size = model.image_size
        self.multimask_output = multimask_output
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics
        self.prompt_encoder = model.sam_prompt_encoder

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1).to(torch.float32)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (point_labels == -1).to(torch.float32)

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels == i).to(torch.float32)

        return point_embedding

    def _batched_mode(self, point_coords: torch.Tensor, point_labels: torch.Tensor):
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        concat_points = (point_coords, point_labels)
        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1  # multi object prediction

        return batched_mode

    def t_embed_masks(self, input_mask: torch.Tensor) -> torch.Tensor:
        mask_embedding = self.prompt_encoder.mask_downscaling(input_mask)
        return mask_embedding

    def mask_postprocessing(self, masks: torch.Tensor) -> torch.Tensor:
        masks = torch.nn.functional.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        high_res_feats_256: torch.Tensor = None,
        high_res_feats_128: torch.Tensor = None,
    ):
        mask_input = None
        sparse_embedding = self._embed_points(point_coords, point_labels)
        if mask_input is None:
            dense_embedding = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(point_coords.shape[0], -1, image_embeddings.shape[0], 64)
        else:
            dense_embedding = self._embed_masks(mask_input)

        batched_mode = self._batched_mode(point_coords, point_labels)

        low_res_masks, iou_pred, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=self.multimask_output,
            repeat_image=batched_mode,
            high_res_features=[high_res_feats_256, high_res_feats_128],
        )

        if self.use_stability_score:
            iou_pred = calculate_stability_score(low_res_masks, self.model.mask_threshold, self.stability_score_offset)

        upscaled_masks = self.mask_postprocessing(low_res_masks)

        if self.return_extra_metrics:
            stability_scores = calculate_stability_score(upscaled_masks, self.model.mask_threshold, self.stability_score_offset)
            areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, low_res_masks, stability_scores, areas

        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)

        return upscaled_masks, iou_pred, low_res_masks
