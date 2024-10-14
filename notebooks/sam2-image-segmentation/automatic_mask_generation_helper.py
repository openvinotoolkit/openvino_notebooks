from sam2.utils.amg import (
    MaskData,
    generate_crop_boxes,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
    calculate_stability_score,
    rle_to_mask,
    batched_mask_to_box,
    mask_to_rle_pytorch,
    is_box_near_crop_edge,
    batch_iterator,
    remove_small_regions,
    build_all_layer_point_grids,
    box_xyxy_to_xywh,
    area_from_rle,
)
from torchvision.ops.boxes import batched_nms, box_area
from typing import Tuple, List, Dict, Any

import torch

from tqdm.notebook import tqdm

import cv2

import numpy as np


from ov_sam2_helper import preprocess_image, postprocess_masks


def draw_anns(image, anns):
    if len(anns) == 0:
        return
    segments_image = image.copy()
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    for ann in tqdm(sorted_anns):
        mask = ann["segmentation"]
        mask_color = np.random.randint(0, 255, size=(1, 1, 3)).astype(np.uint8)
        segments_image[mask] = mask_color
    return cv2.addWeighted(image.astype(np.float32), 0.7, segments_image.astype(np.float32), 0.3, 0.0)


class AutomaticMaskGenerationHelper:
    def __init__(self, resizer, ov_mask_predictor, ov_encoder) -> None:
        self.resizer = resizer
        self.ov_mask_predictor = ov_mask_predictor
        self.ov_encoder = ov_encoder

    def process_batch(
        self,
        image_embedding: np.ndarray,
        high_res_feats_256: np.ndarray,
        high_res_feats_128: np.ndarray,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        iou_thresh,
        mask_threshold,
        stability_score_offset,
        stability_score_thresh,
        normalize=False,
    ) -> MaskData:
        orig_h, orig_w = orig_size

        #     # Run model on this batch
        transformed_points = self.resizer.apply_coords(points, im_size)
        in_points = transformed_points
        in_labels = np.ones(in_points.shape[0], dtype=int)

        inputs = {
            "image_embeddings": image_embedding,
            "high_res_feats_256": high_res_feats_256,
            "high_res_feats_128": high_res_feats_128,
            "point_coords": in_points[:, None, :],
            "point_labels": in_labels[:, None],
        }
        res = self.ov_mask_predictor(inputs)
        masks = postprocess_masks(res[self.ov_mask_predictor.output(0)], orig_size, self.resizer)

        masks = torch.from_numpy(masks)
        iou_preds = torch.from_numpy(res[self.ov_mask_predictor.output(1)])

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
            low_res_masks=res[self.ov_mask_predictor.output(2)],
        )
        del masks

        # Filter by predicted IoU
        if iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > iou_thresh
            data.filter(keep_mask)

        # Calculate and filter by stability score
        data["stability_score"] = calculate_stability_score(data["masks"], mask_threshold, stability_score_offset)
        if stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    def process_crop(
        self,
        image: np.ndarray,
        point_grids,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
        box_nms_thresh: float = 0.7,
        mask_threshold: float = 0.0,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        preprocessed_cropped_im = preprocess_image(cropped_im, self.resizer)
        crop_embeddings = self.ov_encoder(preprocessed_cropped_im)[self.ov_encoder.output(0)]
        high_res_feats_256 = self.ov_encoder(preprocessed_cropped_im)[self.ov_encoder.output(1)]
        high_res_feats_128 = self.ov_encoder(preprocessed_cropped_im)[self.ov_encoder.output(2)]

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(points_per_batch, points_for_image):
            batch_data = self.process_batch(
                crop_embeddings,
                high_res_feats_256,
                high_res_feats_128,
                points,
                cropped_im_size,
                crop_box,
                orig_size,
                pred_iou_thresh,
                mask_threshold,
                stability_score_offset,
                stability_score_thresh,
            )
            data.cat(batch_data)
            del batch_data

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def generate_masks(self, image: np.ndarray, point_grids, crop_n_layers, crop_overlap_ratio, crop_nms_thresh) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(orig_size, crop_n_layers, crop_overlap_ratio)

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self.process_crop(image, point_grids, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        data.to_numpy()
        return data

    def postprocess_small_regions(self, mask_data: MaskData, min_area: int, nms_thresh: float) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def automatic_mask_generation(
        self,
        image: np.ndarray,
        min_mask_region_area: int = 0,
        points_per_side: int = 32,
        crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 1,
        crop_overlap_ratio: float = 512 / 1500,
        box_nms_thresh: float = 0.7,
        crop_nms_thresh: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
        image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
        list(dict(str, any)): A list over records for masks. Each record is
            a dict containing the following keys:
            segmentation (dict(str, any) or np.ndarray): The mask. If
                output_mode='binary_mask', is an array of shape HW. Otherwise,
                is a dictionary containing the RLE.
            bbox (list(float)): The box around the mask, in XYWH format.
            area (int): The area in pixels of the mask.
            predicted_iou (float): The model's own prediction of the mask's
                quality. This is filtered by the pred_iou_thresh parameter.
            point_coords (list(list(float))): The point coordinates input
                to the model to generate this mask.
            stability_score (float): A measure of the mask's quality. This
                is filtered on using the stability_score_thresh parameter.
            crop_box (list(float)): The crop of the image used to generate
                the mask, given in XYWH format.
        """
        point_grids = build_all_layer_point_grids(
            points_per_side,
            crop_n_layers,
            crop_n_points_downscale_factor,
        )

        mask_data = self.generate_masks(image, point_grids, crop_n_layers, crop_overlap_ratio, crop_nms_thresh)

        # Filter small disconnected regions and holes in masks
        if min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                min_mask_region_area,
                max(box_nms_thresh, crop_nms_thresh),
            )

        mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns
