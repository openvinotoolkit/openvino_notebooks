import torch

from sam2.modeling.sam2_base import NO_OBJ_SCORE

from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames

from sam2.utils.amg import calculate_stability_score

from sam2.sam2_video_predictor import SAM2VideoPredictor

import math
import warnings

import torch.nn.functional as F
from torch import Tensor

from sam2.modeling.sam.transformer import sdp_kernel_context


class SamVideoFrameEncoderModel(torch.nn.Module):
    def __init__(self, predictor) -> None:
        super().__init__()
        self.image_encoder = predictor.image_encoder
        self.predictor = predictor

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,
    ):
        backbone_out = self.image_encoder(image)

        if self.predictor.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.predictor.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.predictor.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])

        return (
            backbone_out["backbone_fpn"],
            backbone_out["vision_pos_enc"],
            backbone_out["vision_features"],
        )


class SamVideoMaskPredictorModel(torch.nn.Module):
    def __init__(
        self,
        model,
        multimask_output: bool,
        use_stability_score: bool = False,
        return_extra_metrics: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.img_size = model.image_size
        self.multimask_output = multimask_output
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics

        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder
        self.pred_obj_scores = model.pred_obj_scores
        self.obj_ptr_proj = model.obj_ptr_proj

        self.no_obj_ptr = model.no_obj_ptr
        self.soft_no_obj_ptr = model.soft_no_obj_ptr
        self.fixed_no_obj_ptr = model.fixed_no_obj_ptr

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor, pad: bool = True) -> torch.Tensor:
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

    def _embed_masks(self, input_mask: torch.Tensor) -> torch.Tensor:
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
        backbone_features,
        point_labels=None,
        point_coords=None,
        mask_inputs=None,
        high_res_feats_256=None,
        high_res_feats_128=None,
    ):
        mask_inputs = None
        sparse_embeddings = self._embed_points(point_coords, point_labels)
        if mask_inputs is None:
            dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(point_coords.shape[0], -1, backbone_features.shape[0], 64)
        else:
            dense_embeddings = self._embed_masks(mask_inputs)

        # batched_modes = self._batched_mode(point_coords, point_labels)
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=[high_res_feats_256, high_res_feats_128],
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = torch.nn.functional.interpolate(
            low_res_multimasks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if self.multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device="cpu")
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                # Only hard possible with gt
                assert not self.teacher_force_obj_scores_for_mem
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )


class SamVideoMemoryEncoderModel(torch.nn.Module):
    def __init__(self, predictor) -> None:
        super().__init__()
        self.memory_encoder = predictor.memory_encoder
        self.predictor = predictor

    @torch.no_grad()
    def forward(self, pix_feat=None, mask_for_mem=None, skip_mask_sigmoid=torch.tensor(1)):
        maskmem_out = self.memory_encoder(
            pix_feat,
            mask_for_mem,
            skip_mask_sigmoid=(skip_mask_sigmoid == 1),  # sigmoid already applied
        )

        return maskmem_out["vision_features"], maskmem_out["vision_pos_enc"]


# Matrix version of rotary enc
# https://github.com/facebookresearch/segment-anything-2/issues/186


def get_rotation_matrices(dim, end_x, end_y, theta=10000.0, device=None, dtype=None):

    powers = torch.linspace(0, 1, 1 + (dim // 4), device=device, dtype=dtype)[:-1]
    base_angles = torch.pow(theta, -powers)

    end_x, end_y = int(end_x), int(end_y)
    x_mults = torch.arange(end_x, device=device, dtype=dtype).repeat(end_y)
    y_mults = torch.arange(end_y, device=device, dtype=dtype).repeat_interleave(end_x)
    angles_xy = (torch.outer(mults, base_angles) for mults in (x_mults, y_mults))

    rotmats_list = []
    for angles in angles_xy:
        sterm, cterm = torch.sin(-angles), torch.cos(-angles)
        rotmat = torch.stack(
            [
                torch.stack([cterm, -sterm], dim=-1),
                torch.stack([sterm, cterm], dim=-1),
            ],
            dim=-1,
        )
        rotmats_list.append(rotmat)

    return torch.cat(rotmats_list, dim=1).unsqueeze(0).unsqueeze(0)


def apply_rotary_matenc(xq, xk, rotmats, repeat_freqs_k=False):
    bq, hq, nq, cq = xq.shape
    bk, hk, nk, ck = xk.shape

    q_out = torch.matmul(rotmats, xq.reshape(bq, hq, nq, cq // 2, 2, 1)).flatten(3)
    k_rotmat = rotmats.repeat(1, 1, nk // nq, 1, 1, 1) if repeat_freqs_k else rotmats
    k_out = torch.matmul(k_rotmat, xk.reshape(bk, hk, nk, ck // 2, 2, 1)).flatten(3)

    return q_out, k_out


def matrix_rope_forward(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0) -> Tensor:
    # Input projections
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    # Separate into heads
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)

    # Apply rotary position encoding
    w = h = math.sqrt(q.shape[-2])
    self.freqs_cis = self.freqs_cis.to(q.device)
    if self.freqs_cis.shape[0] != q.shape[-2]:
        self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)

    self.rotmats = self.rotmats.to(q.device)
    if self.rotmats.shape[0] != q.shape[-2]:
        self.rotmats = get_rotation_matrices(dim=self.internal_dim // self.num_heads, end_x=w, end_y=h, theta=self.rope_theta)

    if q.shape[-2] != k.shape[-2]:
        assert self.rope_k_repeat

    num_k_rope = k.size(-2) - num_k_exclude_rope
    q, k[:, :, :num_k_rope] = apply_rotary_matenc(
        q,
        k[:, :, :num_k_rope],
        rotmats=self.rotmats,
        repeat_freqs_k=self.rope_k_repeat,
    )

    dropout_p = self.dropout_p if self.training else 0.0
    # Attention
    try:
        with sdp_kernel_context(dropout_p):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    except Exception as e:
        # Fall back to all kernels if the Flash attention kernel fails
        warnings.warn(
            f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
            f"kernels for scaled_dot_product_attention (which may have a slower speed).",
            category=UserWarning,
            stacklevel=2,
        )
        global ALLOW_ALL_KERNELS
        ALLOW_ALL_KERNELS = True
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    out = self._recombine_heads(out)
    out = self.out_proj(out)

    return out


class SamVideoMemoryAttentionModel(torch.nn.Module):
    def __init__(
        self,
        predictor,
    ) -> None:
        super().__init__()
        self.memory_attention_predictor = predictor.memory_attention

    @torch.no_grad()
    def forward(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: torch.Tensor = None,
        memory_pos: torch.Tensor = None,
        num_obj_ptr_tokens: int = 0,
    ):
        normed_output = self.memory_attention_predictor(
            curr=curr, curr_pos=curr_pos, memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=num_obj_ptr_tokens
        )

        return normed_output


class OVSAM2VideoPredictor(SAM2VideoPredictor):
    def __init__(
        self,
        ov_image_encoder,
        ov_mask_encoder,
        ov_memory_encoder,
        ov_memory_attention_model,
        memory_encoder_out_proj_weight_shape=None,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        **kwargs,
    ) -> None:
        super().__init__(
            fill_hole_area=fill_hole_area,
            non_overlap_masks=non_overlap_masks,
            clear_non_cond_mem_around_input=clear_non_cond_mem_around_input,
            clear_non_cond_mem_for_multi_obj=clear_non_cond_mem_for_multi_obj,
            **kwargs,
        )

        self.ov_image_encoder = ov_image_encoder
        self.ov_mask_encoder = ov_mask_encoder

        self.ov_memory_encoder = ov_memory_encoder
        self.ov_memory_attention_model = ov_memory_attention_model

        if memory_encoder_out_proj_weight_shape is not None:
            self.mem_dim = memory_encoder_out_proj_weight_shape

        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(torch.zeros(self.num_maskmem, 1, 1, self.mem_dim))

    @classmethod
    def from_pretrained(
        cls,
        model_info,
        ov_image_encoder,
        ov_mask_encoder,
        ov_memory_encoder,
        ov_memory_attention_model,
        memory_encoder_out_proj_weight_shape=None,
        apply_postprocessing=True,
    ):

        v_inputs = {
            "sigmoid_scale_for_mem_enc": model_info["model"]["sigmoid_scale_for_mem_enc"],
            "sigmoid_bias_for_mem_enc": model_info["model"]["sigmoid_bias_for_mem_enc"],
            "num_maskmem": model_info["model"]["num_maskmem"],
            "use_obj_ptrs_in_encoder": model_info["model"]["use_obj_ptrs_in_encoder"],
            "only_obj_ptrs_in_the_past_for_eval": model_info["model"]["only_obj_ptrs_in_the_past_for_eval"],
            "add_tpos_enc_to_obj_ptrs": model_info["model"]["add_tpos_enc_to_obj_ptrs"],
            "directly_add_no_mem_embed": model_info["model"]["directly_add_no_mem_embed"],
            "pred_obj_scores": model_info["model"]["pred_obj_scores"],
            "pred_obj_scores_mlp": model_info["model"]["pred_obj_scores_mlp"],
            "fixed_no_obj_ptr": model_info["model"]["fixed_no_obj_ptr"],
            "multimask_min_pt_num": model_info["model"]["multimask_min_pt_num"],
            "multimask_max_pt_num": model_info["model"]["multimask_max_pt_num"],
            "multimask_output_for_tracking": model_info["model"]["multimask_output_for_tracking"],
            "use_mlp_for_obj_ptr_proj": model_info["model"]["use_mlp_for_obj_ptr_proj"],
            "image_size": model_info["model"]["image_size"],
            "use_mask_input_as_output_without_sam": model_info["model"]["use_mask_input_as_output_without_sam"],
            "use_high_res_features_in_sam": model_info["model"]["use_high_res_features_in_sam"],
            "multimask_output_in_sam": model_info["model"]["multimask_output_in_sam"],
            "use_multimask_token_for_obj_ptr": model_info["model"]["use_multimask_token_for_obj_ptr"],
            "iou_prediction_use_sigmoid": model_info["model"]["iou_prediction_use_sigmoid"],
            "compile_image_encoder": False,
            "image_encoder": ov_image_encoder,
            "memory_attention": ov_memory_attention_model,
            "memory_encoder": ov_memory_encoder,
        }

        if apply_postprocessing:
            v_inputs["fill_hole_area"] = 8
            v_inputs["binarize_mask_from_pts_for_mem_enc"] = True

        return cls(
            ov_image_encoder=ov_image_encoder,
            ov_mask_encoder=ov_mask_encoder,
            ov_memory_encoder=ov_memory_encoder,
            ov_memory_attention_model=ov_memory_attention_model,
            memory_encoder_out_proj_weight_shape=memory_encoder_out_proj_weight_shape,
            **v_inputs,
        )

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            enc_image_res = self.ov_image_encoder(image)
            backbone_out = {
                "vision_features": enc_image_res[self.ov_image_encoder.output(6)],
                "vision_pos_enc": [enc_image_res[self.ov_image_encoder.output(i)] for i in range(3, 6)],
                "backbone_fpn": [enc_image_res[self.ov_image_encoder.output(i)] for i in range(0, 3)],
            }
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = torch.from_numpy(feat).expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = torch.from_numpy(pos).expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(frame_idx, cond_outputs, self.max_cond_frames_in_attn)
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with r>1), in which case
            # we take (self.num_maskmem - 2) frames among every r-th frames plus the last frame.
            r = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // r) * r
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // r) * r
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {t: out for t, out in selected_cond_outputs.items() if (t >= frame_idx if track_in_reverse else t <= frame_idx)}
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (abs(frame_idx - t), out["obj_ptr"])
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond_outputs.get(t, None))
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid emtpy memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        ov_memory_attention_out = self.ov_memory_attention_model(
            inputs={
                "curr": current_vision_feats[0],
                "curr_pos": current_vision_pos_embeds[0],
                "memory": memory,
                "memory_pos": memory_pos_embed,
                "num_obj_ptr_tokens": num_obj_ptr_tokens,
            }
        )

        pix_feat_with_mem = torch.from_numpy(ov_memory_attention_out[self.ov_memory_attention_model.output(0)])
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

        return pix_feat_with_mem

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device)
        else:
            high_res_feats_256, high_res_feats_128 = high_res_features
            inputs = {
                "backbone_features": backbone_features,
                "high_res_feats_256": high_res_feats_256,
                "high_res_feats_128": high_res_feats_128,
            }

            mask_inputs = self.mask_downsample(mask_inputs_float)
            if mask_inputs:
                inputs["point_labels"] = mask_inputs["point_labels"]
                inputs["point_coords"] = mask_inputs["point_coords"]

            sam_outputs = self.ov_mask_encoder(inputs=inputs)

            # produce an object pointer using the SAM decoder from the mask input
            # _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
            #     backbone_features=backbone_features,
            #     mask_inputs=self.mask_downsample(mask_inputs_float),
            #     high_res_features=high_res_features,
            # )

            obj_ptr = torch.from_numpy(sam_outputs[self.ov_mask_encoder.output(5)])
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        use_mask_input_as_output_without_sam=False,
        hidden_dim=None,
        num_maskmem=7,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}

        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s) for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])]
        else:
            high_res_features = None
        if mask_inputs is not None and use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            high_res_feats_256, high_res_feats_128 = high_res_features
            inputs = {
                "backbone_features": pix_feat_with_mem,
                "high_res_feats_256": high_res_feats_256,
                "high_res_feats_128": high_res_feats_128,
            }

            if point_inputs:
                inputs["point_labels"] = point_inputs["point_labels"]
                inputs["point_coords"] = point_inputs["point_coords"]

            if mask_inputs:
                inputs["mask_inputs"] = mask_inputs

            sam_outputs = self.ov_mask_encoder(inputs=inputs)

        low_res_masks = torch.from_numpy(sam_outputs[self.ov_mask_encoder.output(3)])
        high_res_masks = torch.from_numpy(sam_outputs[self.ov_mask_encoder.output(4)])
        obj_ptr = torch.from_numpy(sam_outputs[self.ov_mask_encoder.output(5)])

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        if run_mem_encoder and num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        return current_out

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        maskmem_out = self.ov_memory_encoder(
            inputs={
                "pix_feat": pix_feat,
                "mask_for_mem": mask_for_mem,
                "skip_mask_sigmoid": torch.tensor(1),
            }  # sigmoid already applied
        )
        maskmem_features = torch.from_numpy(maskmem_out[self.ov_memory_encoder.output(0)])
        maskmem_pos_enc = [torch.from_numpy(maskmem_out[self.ov_memory_encoder.output(1)])]

        return maskmem_features, maskmem_pos_enc
