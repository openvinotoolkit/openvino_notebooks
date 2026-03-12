# Copyright (c) OpenVINO contributors. All rights reserved.
# Helper module for SAM3 OpenVINO conversion and inference pipeline.

import math
import warnings
from copy import copy
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import openvino as ov


# ============================================================================
# Part 1: RoPE Matrix Replacement (ViT Backbone)
# ============================================================================
# SAM3's ViT uses complex tensor-based RoPE (compute_axial_cis + apply_rotary_enc).
# OpenVINO doesn't support complex tensors, so we replace with matrix multiplication.
# Pattern adapted from SAM2 video segmentation notebook.


def get_vit_rotation_matrices(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
) -> Tensor:
    """
    Pre-compute 2D rotation matrices for ViT RoPE.
    Replaces compute_axial_cis which produces complex tensors.

    Returns rotation matrices of shape (end_x*end_y, dim//2, 2, 2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x = (torch.arange(end_x * end_y) % end_x).float() * scale_pos
    t_y = torch.div(torch.arange(end_x * end_y), end_x, rounding_mode="floor").float() * scale_pos

    angles_x = torch.outer(t_x, freqs)  # (end_x*end_y, dim//4)
    angles_y = torch.outer(t_y, freqs)  # (end_x*end_y, dim//4)

    # Build 2x2 rotation matrices
    rotmats_list = []
    for angles in (angles_x, angles_y):
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        # Each rotation matrix: [[cos, -sin], [sin, cos]]
        rotmat = torch.stack(
            [
                torch.stack([cos_a, -sin_a], dim=-1),
                torch.stack([sin_a, cos_a], dim=-1),
            ],
            dim=-1,
        )  # (N, dim//4, 2, 2)
        rotmats_list.append(rotmat)

    # Concatenate x and y rotations along feature dim
    return torch.cat(rotmats_list, dim=1)  # (N, dim//2, 2, 2)


def apply_rotary_matmul(
    xq: Tensor,
    xk: Tensor,
    rotmats: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary encoding using matrix multiplication instead of complex tensors.
    xq, xk: (B, nHeads, N, headDim) or reshaped
    rotmats: (N, headDim//2, 2, 2)
    """
    bq, hq, nq, cq = xq.shape
    # Reshape to pairs: (B, H, N, dim//2, 2)
    xq_pairs = xq.float().reshape(bq, hq, nq, cq // 2, 2)
    # Apply rotation: rotmats is (N, dim//2, 2, 2), xq_pairs is (B, H, N, dim//2, 2)
    # We need rotmats to be (1, 1, N, dim//2, 2, 2)
    rot = rotmats.unsqueeze(0).unsqueeze(0)  # (1, 1, N, dim//2, 2, 2)
    xq_out = torch.matmul(rot, xq_pairs.unsqueeze(-1)).squeeze(-1)  # (B, H, N, dim//2, 2)
    xq_out = xq_out.flatten(3).to(xq.dtype)  # (B, H, N, dim)

    if xk.shape[-2] == 0:
        return xq_out, xk

    bk, hk, nk, ck = xk.shape
    xk_pairs = xk.float().reshape(bk, hk, nk, ck // 2, 2)
    rot_k = rot
    if nk != nq:
        # Handle repeat_freqs_k case
        r = nk // nq
        rot_k = rot.repeat(1, 1, r, 1, 1, 1)
    xk_out = torch.matmul(rot_k[:, :, :nk], xk_pairs.unsqueeze(-1)).squeeze(-1)
    xk_out = xk_out.flatten(3).to(xk.dtype)

    return xq_out, xk_out


def patch_vit_rope(vit_model):
    """
    Monkey-patch ViT attention blocks to use matrix-based RoPE
    instead of complex tensor-based RoPE.
    """
    for block in vit_model.blocks:
        attn = block.attn
        if not attn.use_rope or attn.freqs_cis is None:
            continue

        # Pre-compute rotation matrices matching the freqs_cis shape
        input_size = attn.input_size
        scale_pos = 1.0
        if attn.rope_interp and attn.rope_pt_size is not None:
            scale_pos = attn.rope_pt_size[0] / input_size[0]

        rotmats = get_vit_rotation_matrices(
            dim=attn.head_dim,
            end_x=input_size[0],
            end_y=input_size[1],
            theta=attn.rope_theta,
            scale_pos=scale_pos,
        )

        # Store rotation matrices as buffer
        attn.register_buffer("rotmats", rotmats)

        # Replace _apply_rope method
        def _apply_rope_matrix(self, q, k):
            if not self.use_rope:
                return q, k
            return apply_rotary_matmul(q, k, self.rotmats.to(q.device))

        import types
        attn._apply_rope = types.MethodType(_apply_rope_matrix, attn)


# ============================================================================
# Part 1b: RoPE Matrix Replacement (Tracker RoPEAttention)
# ============================================================================
# The tracker (SAM2-style) uses sam3.sam.transformer.RoPEAttention
# which also uses complex tensors. Same replacement pattern.


def get_tracker_rotation_matrices(dim, end_x, end_y, theta=10000.0):
    """
    Pre-compute rotation matrices for tracker RoPEAttention.
    Uses the same format as SAM2 video helper.
    """
    powers = torch.linspace(0, 1, 1 + (dim // 4), dtype=torch.float32)[:-1]
    base_angles = torch.pow(theta, -powers)

    end_x, end_y = int(end_x), int(end_y)
    x_mults = torch.arange(end_x, dtype=torch.float32).repeat(end_y)
    y_mults = torch.arange(end_y, dtype=torch.float32).repeat_interleave(end_x)
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


def apply_tracker_rotary_matenc(xq, xk, rotmats, repeat_freqs_k=False):
    """Apply rotary encoding for tracker attention using matrix multiplication."""
    bq, hq, nq, cq = xq.shape
    bk, hk, nk, ck = xk.shape

    q_out = torch.matmul(rotmats, xq.reshape(bq, hq, nq, cq // 2, 2, 1)).flatten(3)
    k_rotmat = rotmats.repeat(1, 1, nk // nq, 1, 1, 1) if repeat_freqs_k else rotmats
    k_out = torch.matmul(k_rotmat, xk.reshape(bk, hk, nk, ck // 2, 2, 1)).flatten(3)

    return q_out, k_out


def tracker_matrix_rope_forward(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0) -> Tensor:
    """Replacement forward for RoPEAttention that uses matrix-based RoPE."""
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)

    w = h = math.sqrt(q.shape[-2])

    if not hasattr(self, "rotmats") or self.rotmats.shape[2] != q.shape[-2]:
        self.rotmats = get_tracker_rotation_matrices(
            dim=self.internal_dim // self.num_heads,
            end_x=w, end_y=h,
            theta=self.rope_theta,
        ).to(q.device)

    num_k_rope = k.size(-2) - num_k_exclude_rope
    q, k[:, :, :num_k_rope] = apply_tracker_rotary_matenc(
        q,
        k[:, :, :num_k_rope],
        rotmats=self.rotmats.to(q.device),
        repeat_freqs_k=self.rope_k_repeat,
    )

    dropout_p = self.dropout_p if self.training else 0.0
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    out = self._recombine_heads(out)
    out = self.out_proj(out)
    return out


def patch_tracker_rope(tracker_model):
    """Monkey-patch tracker RoPEAttention layers to use matrix-based RoPE."""
    import types
    from sam3.sam.transformer import RoPEAttention

    for module in tracker_model.modules():
        if isinstance(module, RoPEAttention):
            # Pre-compute rotation matrices
            feat_sizes = module.feat_sizes
            if feat_sizes is not None:
                rotmats = get_tracker_rotation_matrices(
                    dim=module.internal_dim // module.num_heads,
                    end_x=feat_sizes[0],
                    end_y=feat_sizes[1],
                    theta=module.rope_theta,
                )
                module.register_buffer("rotmats", rotmats)
            module.forward = types.MethodType(tracker_matrix_rope_forward, module)


# ============================================================================
# Part 2: Wrapper Classes for Model Conversion
# ============================================================================


class Sam3ImageEncoderModel(nn.Module):
    """
    Wraps SAM3 ViT+Neck (Sam3DualViTDetNeck) for OpenVINO conversion.
    Input: (1, 3, 1008, 1008) float32
    Output: backbone_fpn features + vision_pos_enc (3 levels after scalp=1)
    """
    def __init__(self, backbone, scalp=1):
        super().__init__()
        self.vision_backbone = backbone
        self.scalp = scalp

    @torch.no_grad()
    def forward(self, image: Tensor):
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(image)

        if self.scalp > 0:
            sam3_features = sam3_features[:-self.scalp]
            sam3_pos = sam3_pos[:-self.scalp]

        # Return flattened: fpn0, fpn1, fpn2, pos0, pos1, pos2
        return tuple(sam3_features) + tuple(sam3_pos)


class Sam3TextEncoderModel(nn.Module):
    """
    Wraps SAM3 VETextEncoder for OpenVINO conversion.
    Input: token_ids (1, seq_len) int64
    Output: text_features (seq_len, 1, 256), text_mask (1, seq_len) bool
    """
    def __init__(self, text_encoder):
        super().__init__()
        self.encoder = text_encoder.encoder
        self.resizer = text_encoder.resizer
        self.context_length = text_encoder.context_length

    @torch.no_grad()
    def forward(self, token_ids: Tensor):
        text_attention_mask = (token_ids != 0).bool()

        _, text_memory = self.encoder(token_ids)
        text_attention_mask_inv = text_attention_mask.ne(1)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = self.resizer(text_memory)

        return text_memory_resized, text_attention_mask_inv


class Sam3TransformerEncoderModel(nn.Module):
    """
    Wraps SAM3 TransformerEncoderFusion for OpenVINO conversion.
    Inputs: image feature (NCHW), image pos enc (NCHW), prompt (S,B,C), prompt_mask (B,S)
    Output: memory, pos_embed, padding_mask, level_start_index, spatial_shapes, valid_ratios
    """
    def __init__(self, encoder, add_pooled_text_to_img_feat=False):
        super().__init__()
        self.encoder = encoder
        # Disable text pooling fusion to simplify tracing
        self.encoder.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat

    @torch.no_grad()
    def forward(
        self,
        img_feat: Tensor,      # (B, C, H, W)
        img_pos: Tensor,        # (B, C, H, W)
        prompt: Tensor,         # (S, B, C)
        prompt_mask: Tensor,    # (B, S)
    ):
        # TransformerEncoderFusion expects lists for multi-level
        result = self.encoder(
            src=[img_feat],
            src_pos=[img_pos],
            prompt=prompt,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=None,  # Already in NCHW format
        )
        return (
            result["memory"],
            result["pos_embed"],
            result["padding_mask"] if result["padding_mask"] is not None else torch.zeros(1),
            result["level_start_index"],
            result["spatial_shapes"],
            result["valid_ratios"],
        )


class Sam3TransformerDecoderModel(nn.Module):
    """
    Wraps SAM3 TransformerDecoder for OpenVINO conversion.
    Includes bbox_embed for box refinement.
    Fixes apply_dac=False, is_instance_prompt=False for inference.
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    @torch.no_grad()
    def forward(
        self,
        memory: Tensor,            # (HW, B, C)
        pos_embed: Tensor,          # (HW, B, C)
        memory_mask: Tensor,        # dummy or actual
        level_start_index: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        prompt: Tensor,             # (S, B, C) - text prompt for cross-attn
        prompt_mask: Tensor,        # (B, S)
    ):
        bs = memory.shape[1]
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        hs, reference_boxes, presence_logits, presence_feats = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=None,
            pos=pos_embed,
            reference_boxes=None,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )

        # hs: (num_layers, nq, bs, d_model)
        # reference_boxes: (num_layers+1, nq, bs, 4)
        # presence_logits: (num_layers, 1, bs) or None
        if presence_logits is None:
            presence_logits = torch.zeros(hs.shape[0], 1, bs)

        return hs, reference_boxes, presence_logits


class Sam3ScoringModel(nn.Module):
    """
    Wraps DotProductScoring for OpenVINO conversion.
    Input: hs (num_layers, B, nq, C), prompt (S, B, C), prompt_mask (B, S)
    Output: pred_logits (num_layers, B, nq, 1)
    """
    def __init__(self, dot_prod_scoring):
        super().__init__()
        self.scorer = dot_prod_scoring

    @torch.no_grad()
    def forward(self, hs: Tensor, prompt: Tensor, prompt_mask: Tensor):
        return self.scorer(hs, prompt, prompt_mask)


class Sam3SegmentationHeadModel(nn.Module):
    """
    Wraps UniversalSegmentationHead for OpenVINO conversion.
    Includes PixelDecoder + MaskPredictor + cross_attend_prompt + instance_seg_head.
    """
    def __init__(self, seg_head):
        super().__init__()
        self.seg_head = seg_head

    @torch.no_grad()
    def forward(
        self,
        fpn0: Tensor,          # (B, C, H0, W0) - highest res
        fpn1: Tensor,          # (B, C, H1, W1)
        fpn2: Tensor,          # (B, C, H2, W2) - lowest res
        obj_queries: Tensor,   # (B, nq, C) - decoder output
        encoder_hidden_states: Tensor,  # (HW, B, C)
        prompt: Tensor,        # (S, B, C)
        prompt_mask: Tensor,   # (B, S)
    ):
        backbone_feats = [fpn0, fpn1, fpn2]
        image_ids = torch.tensor([0], device=fpn0.device)
        result = self.seg_head(
            backbone_feats=backbone_feats,
            obj_queries=obj_queries.unsqueeze(0),  # add layer dim for last-layer only
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        return result["pred_masks"]


# ============================================================================
# Part 2b: Tracker Wrapper Classes (SAM1 task / Video)
# ============================================================================


class Sam3SAM2PromptEncoderModel(nn.Module):
    """Wraps SAM2-style prompt encoder from Sam3TrackerBase."""
    def __init__(self, prompt_encoder, image_size):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.image_size = image_size

    @torch.no_grad()
    def forward(
        self,
        point_coords: Tensor,   # (B, N, 2)
        point_labels: Tensor,   # (B, N)
        has_box: Tensor,        # scalar indicator
    ):
        # Add 0.5 offset and normalize
        point_coords_norm = (point_coords + 0.5) / self.image_size

        # Encode points
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords_norm)
        point_labels_expanded = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels_expanded != -1).float()
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (point_labels_expanded == -1).float()

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels_expanded == i).float()

        sparse_embeddings = point_embedding
        dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            point_coords.shape[0], -1,
            self.prompt_encoder.image_embedding_size[0],
            self.prompt_encoder.image_embedding_size[1],
        )

        return sparse_embeddings, dense_embeddings


class Sam3SAM2MaskDecoderModel(nn.Module):
    """Wraps SAM2-style mask decoder from Sam3TrackerBase."""
    def __init__(self, model, multimask_output=True):
        super().__init__()
        self.mask_decoder = model.sam_mask_decoder
        self.model = model
        self.multimask_output = multimask_output
        self.img_size = model.image_size

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: Tensor,       # (B, C, H, W)
        high_res_feats_0: Tensor,       # (B, C, H0, W0)
        high_res_feats_1: Tensor,       # (B, C, H1, W1)
        sparse_embeddings: Tensor,      # (B, N, C)
        dense_embeddings: Tensor,       # (B, C, H, W)
    ):
        low_res_masks, iou_pred, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.mask_decoder.pe_layer.forward(image_embeddings.shape[-2:]).unsqueeze(0),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=[high_res_feats_0, high_res_feats_1],
        )

        # Upscale masks to image size
        high_res_masks = F.interpolate(
            low_res_masks, (self.img_size, self.img_size),
            mode="bilinear", align_corners=False
        )

        return low_res_masks, high_res_masks, iou_pred


class Sam3MemoryEncoderModel(nn.Module):
    """Wraps SimpleMaskEncoder for video memory encoding."""
    def __init__(self, memory_encoder):
        super().__init__()
        self.memory_encoder = memory_encoder

    @torch.no_grad()
    def forward(self, pix_feat: Tensor, mask_for_mem: Tensor, skip_mask_sigmoid: Tensor):
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem,
            skip_mask_sigmoid=(skip_mask_sigmoid == 1),
        )
        return maskmem_out["vision_features"], maskmem_out["vision_pos_enc"]


class Sam3MemoryAttentionModel(nn.Module):
    """Wraps TransformerEncoderCrossAttention for video memory attention."""
    def __init__(self, memory_attention):
        super().__init__()
        self.memory_attention = memory_attention

    @torch.no_grad()
    def forward(
        self,
        curr: Tensor,
        memory: Tensor,
        curr_pos: Tensor,
        memory_pos: Tensor,
        num_obj_ptr_tokens: Tensor,
    ):
        return self.memory_attention(
            curr=curr, curr_pos=curr_pos,
            memory=memory, memory_pos=memory_pos,
            num_obj_ptr_tokens=int(num_obj_ptr_tokens.item()),
        )


# ============================================================================
# Part 3: Model Conversion Utilities
# ============================================================================


def convert_and_save_model(
    wrapper_model: nn.Module,
    example_input,
    save_path: str,
    model_name: str,
):
    """Convert a PyTorch model wrapper to OpenVINO IR and save."""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"  [skip] {model_name} already exists at {save_path}")
        return ov.Core().read_model(save_path)

    print(f"  Converting {model_name}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        ov_model = ov.convert_model(wrapper_model, example_input=example_input)

    ov.save_model(ov_model, save_path)
    print(f"  Saved {model_name} to {save_path}")
    return ov_model


# ============================================================================
# Part 4: OpenVINO Pipeline Classes
# ============================================================================


class OVSam3Processor:
    """
    Drop-in replacement for Sam3Processor using OpenVINO compiled models.
    Mirrors the Sam3Processor API: set_image, set_text_prompt, add_geometric_prompt.
    """

    def __init__(
        self,
        original_model,  # PyTorch Sam3Image model (for geometry encoder, etc.)
        ov_image_encoder,
        ov_text_encoder,
        ov_transformer_encoder,
        ov_transformer_decoder,
        ov_scoring,
        ov_seg_head,
        tokenizer,
        resolution=1008,
        confidence_threshold=0.5,
    ):
        self.original_model = original_model
        self.ov_image_encoder = ov_image_encoder
        self.ov_text_encoder = ov_text_encoder
        self.ov_transformer_encoder = ov_transformer_encoder
        self.ov_transformer_decoder = ov_transformer_decoder
        self.ov_scoring = ov_scoring
        self.ov_seg_head = ov_seg_head
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold

        from torchvision.transforms import v2
        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        from sam3.model.data_misc import FindStage
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], dtype=torch.long),
            text_ids=torch.tensor([0], dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Encode image using OV image encoder."""
        import PIL
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        from torchvision.transforms import v2
        img_tensor = v2.functional.to_image(image)
        img_tensor = self.transform(img_tensor).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width

        # Run OV image encoder
        ov_result = self.ov_image_encoder(img_tensor.numpy())

        # Parse outputs: fpn0, fpn1, fpn2, pos0, pos1, pos2
        n_outputs = len(ov_result)
        n_levels = n_outputs // 2
        backbone_fpn = [torch.from_numpy(ov_result[i].data) for i in range(n_levels)]
        vision_pos_enc = [torch.from_numpy(ov_result[n_levels + i].data) for i in range(n_levels)]

        state["backbone_out"] = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": vision_pos_enc,
            "vision_features": backbone_fpn[-1],
        }

        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Encode text using OV text encoder."""
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        # Tokenize
        tokenized = self.tokenizer([prompt], context_length=32)
        token_ids = tokenized.numpy()

        # Run OV text encoder
        ov_result = self.ov_text_encoder(token_ids)
        text_memory_resized = torch.from_numpy(ov_result[0].data)
        text_attention_mask = torch.from_numpy(ov_result[1].data)

        state["backbone_out"]["language_features"] = text_memory_resized
        state["backbone_out"]["language_mask"] = text_attention_mask
        state["backbone_out"]["language_embeds"] = text_memory_resized  # simplified

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.original_model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Add a box prompt and run inference."""
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_geometric_prompt")

        if "language_features" not in state["backbone_out"]:
            # Set dummy text
            tokenized = self.tokenizer(["visual"], context_length=32)
            ov_result = self.ov_text_encoder(tokenized.numpy())
            state["backbone_out"]["language_features"] = torch.from_numpy(ov_result[0].data)
            state["backbone_out"]["language_mask"] = torch.from_numpy(ov_result[1].data)
            state["backbone_out"]["language_embeds"] = torch.from_numpy(ov_result[0].data)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.original_model._get_dummy_prompt()

        from sam3.model.geometry_encoders import Prompt
        boxes = torch.tensor(box, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Remove all prompts and results."""
        if "backbone_out" in state:
            for key in ["language_features", "language_mask", "language_embeds"]:
                state["backbone_out"].pop(key, None)
        for key in ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]:
            state.pop(key, None)

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        """Run the full grounding pipeline using OV models."""
        backbone_out = state["backbone_out"]
        geometric_prompt = state["geometric_prompt"]

        # Step 1: Encode prompt (geometry encoder stays in PyTorch)
        prompt, prompt_mask, backbone_out = self.original_model._encode_prompt(
            backbone_out, self.find_stage, geometric_prompt.clone()
        )

        # Step 2: Get image features
        feat_tuple = self.original_model._get_img_feats(backbone_out, self.find_stage.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        # Reshape image features from seq-first to NCHW for encoder
        bs = img_feats[0].shape[1]
        h, w = vis_feat_sizes[0]
        img_feat_nchw = img_feats[0].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
        img_pos_nchw = img_pos_embeds[0].reshape(h, w, bs, -1).permute(2, 3, 0, 1)

        # Handle text pooling fusion manually if needed
        if self.original_model.transformer.encoder.add_pooled_text_to_img_feat:
            from sam3.model.encoder import pool_text_feat
            pooled = pool_text_feat(
                prompt, prompt_mask,
                self.original_model.transformer.encoder.pool_text_with_mask
            )
            pooled = self.original_model.transformer.encoder.text_pooling_proj(pooled)[..., None, None]
            img_feat_nchw = img_feat_nchw + pooled

        # Step 3: Run OV transformer encoder
        prompt_bf = prompt.transpose(0, 1)  # batch-first for encoder internal
        enc_result = self.ov_transformer_encoder({
            "img_feat": img_feat_nchw.numpy(),
            "img_pos": img_pos_nchw.numpy(),
            "prompt": prompt.numpy(),
            "prompt_mask": prompt_mask.numpy(),
        })

        memory = torch.from_numpy(enc_result[0].data)
        pos_embed = torch.from_numpy(enc_result[1].data)
        padding_mask = torch.from_numpy(enc_result[2].data) if enc_result[2].data.size > 1 else None
        level_start_index = torch.from_numpy(enc_result[3].data)
        spatial_shapes = torch.from_numpy(enc_result[4].data)
        valid_ratios = torch.from_numpy(enc_result[5].data)

        # Step 4: Run OV transformer decoder
        dec_result = self.ov_transformer_decoder({
            "memory": memory.numpy(),
            "pos_embed": pos_embed.numpy(),
            "memory_mask": torch.zeros(1).numpy(),
            "level_start_index": level_start_index.numpy(),
            "spatial_shapes": spatial_shapes.numpy(),
            "valid_ratios": valid_ratios.numpy(),
            "prompt": prompt.numpy(),
            "prompt_mask": prompt_mask.numpy(),
        })

        hs = torch.from_numpy(dec_result[0].data)  # (num_layers, nq, bs, C)
        reference_boxes = torch.from_numpy(dec_result[1].data)
        presence_logits = torch.from_numpy(dec_result[2].data)

        # Transpose to batch-first for scoring
        hs_bf = hs.transpose(1, 2)  # (num_layers, bs, nq, C)
        reference_boxes_bf = reference_boxes.transpose(1, 2)

        # Step 5: Run OV scoring
        score_result = self.ov_scoring({
            "hs": hs_bf.numpy(),
            "prompt": prompt.numpy(),
            "prompt_mask": prompt_mask.numpy(),
        })
        pred_logits = torch.from_numpy(score_result[0].data)

        # Step 6: Compute boxes from decoder output
        from sam3.model.model_misc import inverse_sigmoid
        from sam3.model.box_ops import box_cxcywh_to_xyxy

        bbox_embed = self.original_model.transformer.decoder.bbox_embed
        anchor_offsets = bbox_embed(self.original_model.transformer.decoder.norm(hs_bf[-1]))
        ref_inv = inverse_sigmoid(reference_boxes_bf[-1])
        pred_boxes = (ref_inv + anchor_offsets).sigmoid()
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)

        # Step 7: Run OV segmentation head
        fpn = backbone_out["backbone_fpn"]
        seg_result = self.ov_seg_head({
            "fpn0": fpn[0].numpy(),
            "fpn1": fpn[1].numpy(),
            "fpn2": fpn[2].numpy(),
            "obj_queries": hs_bf[-1].numpy(),
            "encoder_hidden_states": memory.numpy(),
            "prompt": prompt.numpy(),
            "prompt_mask": prompt_mask.numpy(),
        })
        pred_masks = torch.from_numpy(seg_result[0].data)

        # Step 8: Post-process
        out_logits = pred_logits[-1]  # last layer
        presence_score = presence_logits[-1].unsqueeze(-1).sigmoid() if presence_logits.numel() > 1 else torch.ones_like(out_logits)
        out_probs = (out_logits.sigmoid() * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = pred_masks[keep]
        out_bbox = pred_boxes[keep]

        from sam3.model import box_ops
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        boxes = boxes * scale_fct[None, :]

        from sam3.model.data_misc import interpolate
        out_masks = interpolate(
            out_masks.unsqueeze(1), (img_h, img_w),
            mode="bilinear", align_corners=False
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state


class OVSam3InteractiveImagePredictor:
    """
    Drop-in replacement for SAM3InteractiveImagePredictor using OV models.
    Used for SAM1-style point/box prompting.
    """
    def __init__(
        self,
        original_predictor,  # SAM3InteractiveImagePredictor
        ov_image_encoder,
        ov_prompt_encoder,
        ov_mask_decoder,
        image_size=1008,
    ):
        self.original = original_predictor
        self.ov_image_encoder = ov_image_encoder
        self.ov_prompt_encoder = ov_prompt_encoder
        self.ov_mask_decoder = ov_mask_decoder
        self.image_size = image_size
        self._transforms = original_predictor._transforms
        self._features = None
        self._orig_hw = None
        self._bb_feat_sizes = original_predictor._bb_feat_sizes
        self.mask_threshold = original_predictor.mask_threshold

    @torch.no_grad()
    def set_image(self, image):
        """Encode image for point/box prompting."""
        if isinstance(image, np.ndarray):
            self._orig_hw = [image.shape[:2]]
        else:
            w, h = image.size
            self._orig_hw = [(h, w)]

        input_image = self._transforms(image)
        input_image = input_image[None, ...]

        # Run OV image encoder (shared with detector)
        ov_result = self.ov_image_encoder(input_image.numpy())
        n_outputs = len(ov_result)
        n_levels = n_outputs // 2

        backbone_fpn = [torch.from_numpy(ov_result[i].data) for i in range(n_levels)]
        vision_pos_enc = [torch.from_numpy(ov_result[n_levels + i].data) for i in range(n_levels)]

        # Process through SAM2-style feature preparation
        # Need to use conv_s0, conv_s1 from the original model's mask decoder
        model = self.original.model
        if hasattr(model, 'sam_mask_decoder'):
            if hasattr(model.sam_mask_decoder, 'conv_s0'):
                backbone_fpn[0] = model.sam_mask_decoder.conv_s0(backbone_fpn[0])
            if hasattr(model.sam_mask_decoder, 'conv_s1'):
                backbone_fpn[1] = model.sam_mask_decoder.conv_s1(backbone_fpn[1])

        self._features = {
            "image_embed": backbone_fpn[-1],
            "high_res_feats": backbone_fpn[:-1],
        }

    @torch.no_grad()
    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
        normalize_coords=True,
    ):
        """Run prediction with point/box prompts using OV models."""
        assert self._features is not None, "Must call set_image first"

        # Prepare prompts (use original model's method if available)
        if point_coords is not None:
            point_coords = torch.as_tensor(point_coords, dtype=torch.float32)
            point_labels = torch.as_tensor(point_labels, dtype=torch.int32)

            if normalize_coords:
                orig_h, orig_w = self._orig_hw[0]
                point_coords = point_coords.clone()
                point_coords[..., 0] = point_coords[..., 0] / orig_w * self.image_size
                point_coords[..., 1] = point_coords[..., 1] / orig_h * self.image_size

        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float32)
            if normalize_coords:
                orig_h, orig_w = self._orig_hw[0]
                box = box.clone()
                box[..., 0] = box[..., 0] / orig_w * self.image_size
                box[..., 1] = box[..., 1] / orig_h * self.image_size
                box[..., 2] = box[..., 2] / orig_w * self.image_size
                box[..., 3] = box[..., 3] / orig_h * self.image_size

        # Combine point and box prompts
        if box is not None:
            box_corners = box.reshape(-1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32)
            if point_coords is not None:
                point_coords = torch.cat([point_coords, box_corners], dim=0) if point_coords.dim() == 2 else torch.cat([point_coords.squeeze(0), box_corners], dim=0)
                point_labels = torch.cat([point_labels.flatten(), box_labels], dim=0)
            else:
                point_coords = box_corners
                point_labels = box_labels

        if point_coords.dim() == 2:
            point_coords = point_coords.unsqueeze(0)
            point_labels = point_labels.unsqueeze(0)

        # Add padding point
        padding_point = torch.zeros((point_coords.shape[0], 1, 2))
        padding_label = -torch.ones((point_labels.shape[0], 1), dtype=torch.int32)
        concat_coords = torch.cat([point_coords, padding_point], dim=1)
        concat_labels = torch.cat([point_labels, padding_label], dim=1)

        # Run OV prompt encoder
        has_box = torch.tensor(1 if box is not None else 0)
        enc_result = self.ov_prompt_encoder({
            "point_coords": concat_coords.numpy(),
            "point_labels": concat_labels.numpy(),
            "has_box": has_box.numpy(),
        })
        sparse_embeddings = torch.from_numpy(enc_result[0].data)
        dense_embeddings = torch.from_numpy(enc_result[1].data)

        # Run OV mask decoder
        image_embed = self._features["image_embed"]
        high_res_feats = self._features["high_res_feats"]

        dec_result = self.ov_mask_decoder({
            "image_embeddings": image_embed.numpy(),
            "high_res_feats_0": high_res_feats[0].numpy(),
            "high_res_feats_1": high_res_feats[1].numpy(),
            "sparse_embeddings": sparse_embeddings,
            "dense_embeddings": dense_embeddings,
        })

        low_res_masks = torch.from_numpy(dec_result[0].data)
        high_res_masks = torch.from_numpy(dec_result[1].data)
        iou_pred = torch.from_numpy(dec_result[2].data)

        # Post-process to original image size
        orig_h, orig_w = self._orig_hw[0]
        masks = F.interpolate(
            high_res_masks, (orig_h, orig_w),
            mode="bilinear", align_corners=False,
        )

        if not return_logits:
            masks = masks > self.mask_threshold

        return masks.squeeze(0).numpy(), iou_pred.squeeze(0).numpy(), low_res_masks.squeeze(0).numpy()


# ============================================================================
# Part 5: Visualization Utilities
# ============================================================================

def show_masks_on_image(image, masks, boxes=None, scores=None, alpha=0.5):
    """Display masks overlaid on an image."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    if masks is not None and len(masks) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(masks), 10)))
        for i, mask in enumerate(masks):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask = mask.squeeze()
            color = colors[i % len(colors)]
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask > 0.5] = [*color[:3], alpha]
            plt.imshow(colored_mask)

    if boxes is not None:
        for i, box in enumerate(boxes):
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            x0, y0, x1, y1 = box
            rect = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            plt.gca().add_patch(rect)
            if scores is not None:
                score = scores[i] if isinstance(scores[i], float) else scores[i].item()
                plt.text(x0, y0 - 5, f"{score:.2f}", color='green', fontsize=10)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def compare_masks(pt_masks, ov_masks, title="Mask Comparison"):
    """Compare PyTorch and OpenVINO masks, compute IoU."""
    import matplotlib.pyplot as plt

    if isinstance(pt_masks, torch.Tensor):
        pt_masks = pt_masks.cpu().numpy()
    if isinstance(ov_masks, torch.Tensor):
        ov_masks = ov_masks.cpu().numpy()

    pt_binary = (pt_masks > 0.5).astype(np.float32)
    ov_binary = (ov_masks > 0.5).astype(np.float32)

    # Compute IoU
    intersection = (pt_binary * ov_binary).sum()
    union = ((pt_binary + ov_binary) > 0).astype(np.float32).sum()
    iou = intersection / max(union, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pt_binary.squeeze(), cmap='gray')
    axes[0].set_title("PyTorch")
    axes[0].axis("off")

    axes[1].imshow(ov_binary.squeeze(), cmap='gray')
    axes[1].set_title("OpenVINO")
    axes[1].axis("off")

    diff = np.abs(pt_binary - ov_binary).squeeze()
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f"Difference (IoU={iou:.4f})")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return iou


def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
