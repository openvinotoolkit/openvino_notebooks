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

    # Build 2x2 rotation matrices: [[cos, -sin], [sin, cos]]
    # Complex multiplication (a+bi)(cosθ+i·sinθ) = (a·cosθ-b·sinθ) + i(a·sinθ+b·cosθ)
    # Equivalent to matrix: [[cos,-sin],[sin,cos]] @ [a,b]^T
    # Use dim=-2 to stack row vectors as ROWS (not columns)
    rotmats_list = []
    for angles in (angles_x, angles_y):
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        rotmat = torch.stack(
            [
                torch.stack([cos_a, -sin_a], dim=-1),  # row 0: [cos, -sin]
                torch.stack([sin_a, cos_a], dim=-1),  # row 1: [sin, cos]
            ],
            dim=-2,
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
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        # Same rotation matrix as ViT: [[cos, -sin], [sin, cos]]
        # Use dim=-2 to stack row vectors as ROWS
        rotmat = torch.stack(
            [
                torch.stack([cos_a, -sin_a], dim=-1),  # row 0
                torch.stack([sin_a, cos_a], dim=-1),  # row 1
            ],
            dim=-2,
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
            end_x=w,
            end_y=h,
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
    Pass model.backbone.vision_backbone (the Sam3DualViTDetNeck) directly.
    Input: (1, 3, 1008, 1008) float32
    Output: sam3_fpn + sam3_pos (3 levels after scalp=1) + sam2_fpn + sam2_pos (3 levels after scalp=1)
    Total outputs: 12 tensors (6 sam3 + 6 sam2) when sam2 neck exists, else 6 sam3
    """

    def __init__(self, neck, scalp=1):
        super().__init__()
        self.neck = neck
        self.scalp = scalp
        self.has_sam2 = neck.sam2_convs is not None

    @torch.no_grad()
    def forward(self, image: Tensor):
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.neck(image)

        if self.scalp > 0:
            sam3_features = sam3_features[: -self.scalp]
            sam3_pos = sam3_pos[: -self.scalp]
            if sam2_features is not None:
                sam2_features = sam2_features[: -self.scalp]
                sam2_pos = sam2_pos[: -self.scalp]

        # Return: sam3_fpn0..2, sam3_pos0..2, [sam2_fpn0..2, sam2_pos0..2]
        result = tuple(sam3_features) + tuple(sam3_pos)
        if sam2_features is not None:
            result = result + tuple(sam2_features) + tuple(sam2_pos)
        return result


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
    Inputs: image feature (HW, B, C) seq-first, image pos enc (HW, B, C), prompt (S,B,C), prompt_mask (B,S)
    Also takes feat_h, feat_w as scalar tensors to reconstruct feat_sizes.
    Output: memory, pos_embed, padding_mask, level_start_index, spatial_shapes, valid_ratios
    """

    def __init__(self, encoder, feat_h=72, feat_w=72):
        super().__init__()
        self.encoder = encoder
        self.feat_h = feat_h
        self.feat_w = feat_w

    @torch.no_grad()
    def forward(
        self,
        img_feat: Tensor,  # (HW, B, C) seq-first
        img_pos: Tensor,  # (HW, B, C) seq-first
        prompt: Tensor,  # (S, B, C)
        prompt_mask: Tensor,  # (B, S)
    ):
        feat_sizes = [(self.feat_h, self.feat_w)]
        result = self.encoder(
            src=[img_feat],
            src_pos=[img_pos],
            prompt=prompt,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=feat_sizes,
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
        memory: Tensor,  # (HW, B, C)
        pos_embed: Tensor,  # (HW, B, C)
        memory_mask: Tensor,  # dummy or actual
        level_start_index: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        prompt: Tensor,  # (S, B, C) - text prompt for cross-attn
        prompt_mask: Tensor,  # (B, S)
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
    Wraps DotProductScoring + bbox_embed + decoder_norm for OpenVINO conversion.
    Input: hs (num_layers, B, nq, C) batch-first, reference_boxes (num_layers+1, B, nq, 4),
           prompt (S, B, C), prompt_mask (B, S)
    Output: pred_logits (num_layers, B, nq, 1), pred_boxes (B, nq, 4) cxcywh [0,1]
    """

    def __init__(self, dot_prod_scoring, bbox_embed=None, decoder_norm=None):
        super().__init__()
        self.scorer = dot_prod_scoring
        self.bbox_embed = bbox_embed
        self.decoder_norm = decoder_norm

    @torch.no_grad()
    def forward(self, hs: Tensor, reference_boxes: Tensor, prompt: Tensor, prompt_mask: Tensor):
        pred_logits = self.scorer(hs, prompt, prompt_mask)

        # Box refinement via bbox_embed + decoder_norm
        last_hs = hs[-1]  # (bs, nq, C)
        anchor_offsets = self.bbox_embed(self.decoder_norm(last_hs))  # (bs, nq, 4)
        ref_last = reference_boxes[-1]  # (bs, nq, 4)
        ref_clamped = ref_last.clamp(min=0, max=1)
        ref_inv = torch.log(ref_clamped.clamp(min=1e-3) / (1 - ref_clamped).clamp(min=1e-3))
        pred_boxes = (ref_inv + anchor_offsets).sigmoid()  # (bs, nq, 4) cxcywh [0,1]

        return pred_logits, pred_boxes


class Sam3DecoderWithScoringModel(nn.Module):
    """
    Merged Transformer Decoder + Scoring Head + Box Refinement for OpenVINO conversion.
    Combines Sam3TransformerDecoderModel, Sam3ScoringModel, bbox_embed, and decoder_norm
    into a single OV model, eliminating all PyTorch post-processing steps.

    Outputs: hs (batch-first), reference_boxes (batch-first), presence_logits,
             pred_logits, pred_boxes (final refined boxes in cxcywh [0,1])
    """

    def __init__(self, decoder, dot_prod_scoring, bbox_embed, decoder_norm):
        super().__init__()
        self.decoder = decoder
        self.scorer = dot_prod_scoring
        self.bbox_embed = bbox_embed
        self.decoder_norm = decoder_norm

    @torch.no_grad()
    def forward(
        self,
        memory: Tensor,  # (HW, B, C)
        pos_embed: Tensor,  # (HW, B, C)
        memory_mask: Tensor,  # dummy or actual
        level_start_index: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        prompt: Tensor,  # (S, B, C)
        prompt_mask: Tensor,  # (B, S)
    ):
        bs = memory.shape[1]
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        hs, reference_boxes, presence_logits, _ = self.decoder(
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

        if presence_logits is None:
            presence_logits = torch.zeros(hs.shape[0], 1, bs)

        # Transpose to batch-first for scoring
        hs_bf = hs.transpose(1, 2)  # (num_layers, bs, nq, C)
        reference_boxes_bf = reference_boxes.transpose(1, 2)  # (num_layers+1, bs, nq, 4)

        # Run scoring
        pred_logits = self.scorer(hs_bf, prompt, prompt_mask)

        # Box refinement via bbox_embed + decoder_norm (eliminates PyTorch post-processing)
        last_hs = hs_bf[-1]  # (bs, nq, C)
        anchor_offsets = self.bbox_embed(self.decoder_norm(last_hs))  # (bs, nq, 4)
        ref_last = reference_boxes_bf[-1]  # (bs, nq, 4)
        # inverse_sigmoid
        ref_clamped = ref_last.clamp(min=0, max=1)
        ref_inv = torch.log(ref_clamped.clamp(min=1e-3) / (1 - ref_clamped).clamp(min=1e-3))
        pred_boxes = (ref_inv + anchor_offsets).sigmoid()  # (bs, nq, 4) cxcywh [0,1]

        return hs_bf, reference_boxes_bf, presence_logits, pred_logits, pred_boxes


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
        fpn0: Tensor,  # (B, C, H0, W0) - highest res
        fpn1: Tensor,  # (B, C, H1, W1)
        fpn2: Tensor,  # (B, C, H2, W2) - lowest res
        obj_queries: Tensor,  # (B, nq, C) - decoder output
        encoder_hidden_states: Tensor,  # (HW, B, C)
        prompt: Tensor,  # (S, B, C)
        prompt_mask: Tensor,  # (B, S)
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


class Sam3GeometryEncoderForOV(nn.Module):
    """
    OV-conversion-friendly geometry encoder wrapper (batch_size=1 only).
    Uses tensor-format roi_align (N,5) instead of list-format for clean OV tracing.
    Otherwise identical logic to Sam3GeometryEncoderModel.
    """

    def __init__(self, geometry_encoder, feat_h=72, feat_w=72):
        super().__init__()
        self.geo = geometry_encoder
        self.feat_h = feat_h
        self.feat_w = feat_w

    @torch.no_grad()
    def forward(
        self,
        box_embeddings: Tensor,  # (N, 1, 4)
        box_mask: Tensor,  # (1, N) bool
        box_labels: Tensor,  # (N, 1) int64
        img_feat: Tensor,  # (HW, 1, C)
        img_pos: Tensor,  # (HW, 1, C)
    ):
        import torchvision.ops
        from sam3.model.box_ops import box_cxcywh_to_xyxy

        geo = self.geo
        n_boxes = box_embeddings.shape[0]
        bs = box_embeddings.shape[1]
        H = self.feat_h
        W = self.feat_w

        boxes = box_embeddings
        boxes_embed = None

        if geo.boxes_direct_project is not None:
            proj = geo.boxes_direct_project(boxes)
            boxes_embed = proj

        if geo.boxes_pool_project is not None:
            img_normed = geo.img_pre_norm(img_feat)
            img_nchw = img_normed.permute(1, 2, 0).view(bs, -1, H, W)

            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            scale = torch.tensor([W, H, W, H], dtype=boxes.dtype, device=boxes.device).view(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale

            # Tensor-format roi_align: (K, 5) = [batch_idx, x1, y1, x2, y2]
            boxes_2d = boxes_xyxy[:, 0, :]  # (N, 4)
            batch_idx = torch.zeros(n_boxes, 1, dtype=boxes.dtype, device=boxes.device)
            boxes_with_batch = torch.cat([batch_idx, boxes_2d], dim=1)  # (N, 5)

            sampled = torchvision.ops.roi_align(img_nchw, boxes_with_batch, geo.roi_size)
            proj = geo.boxes_pool_project(sampled)
            proj = proj.view(bs, n_boxes, -1).transpose(0, 1)

            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        if geo.boxes_pos_enc_project is not None:
            cx, cy, w, h = boxes.unbind(-1)
            enc = geo.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
            enc = enc.view(n_boxes, bs, enc.shape[-1])
            proj = geo.boxes_pos_enc_project(enc)

            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        label_embed = geo.label_embed(box_labels.long())
        boxes_embed = label_embed + boxes_embed

        cls = geo.cls_embed.weight.view(1, 1, -1).expand(1, bs, -1)
        cls_mask = torch.zeros(bs, 1, dtype=box_mask.dtype, device=box_mask.device)

        combined = torch.cat([boxes_embed, cls], dim=0)
        combined_mask = torch.cat([box_mask, cls_mask], dim=1)

        if geo.final_proj is not None:
            combined = geo.norm(geo.final_proj(combined))

        if geo.encode is not None:
            for lay in geo.encode:
                combined = lay(
                    tgt=combined,
                    memory=img_feat,
                    tgt_key_padding_mask=combined_mask,
                    pos=img_pos,
                )
            combined = geo.encode_norm(combined)

        return combined, combined_mask


class Sam3GeoProjectionsForOV(nn.Module):
    """
    Merged weighted projection layers for SAM3 geometry encoder.
    Contains boxes_direct_project, boxes_pos_enc_project, boxes_pool_project,
    and label_embed — NO control flow, NO roi_align, NO sinusoidal pos_enc.

    Non-weighted ops (roi_align, pos_enc.encode_boxes, img_pre_norm, CLS concat)
    are handled in Python by OVSam3Processor.

    Inputs:
        boxes:         (N, 1, 4)  normalized cxcywh
        pos_features:  (N, 1, d_pos)  sinusoidal pos encoding (computed in Python)
        roi_features:  (N, C, roi_h, roi_w)  roi_align output (computed in Python)
        labels:        (N, 1)  int64  0=neg, 1=pos

    Output:
        embed: (N, 1, d_model)  box embeddings (sum of all projections + label)
    """

    def __init__(self, geo):
        super().__init__()
        assert geo.boxes_direct_project is not None, "boxes_direct_project required"
        assert geo.boxes_pool_project is not None, "boxes_pool_project required"
        assert geo.boxes_pos_enc_project is not None, "boxes_pos_enc_project required"

        self.direct_proj = geo.boxes_direct_project
        self.pos_enc_proj = geo.boxes_pos_enc_project
        self.pool_proj = geo.boxes_pool_project
        self.label_embed = geo.label_embed

    @torch.no_grad()
    def forward(
        self,
        boxes: Tensor,  # (N, 1, 4)
        pos_features: Tensor,  # (N, 1, d_pos)
        roi_features: Tensor,  # (N, C, roi_h, roi_w)
        labels: Tensor,  # (N, 1) int64
    ):
        direct = self.direct_proj(boxes)  # (N, 1, C)
        pos = self.pos_enc_proj(pos_features)  # (N, 1, C)
        pool = self.pool_proj(roi_features)  # (N, C, 1, 1)
        pool = pool.squeeze(-1).squeeze(-1).unsqueeze(1)  # (N, 1, C)
        lab = self.label_embed(labels.long())  # (N, 1, C)
        return direct + pos + pool + lab


class Sam3GeoCrossAttnForOV(nn.Module):
    """
    Cross-attention encoder for geometry features.
    Contains cls_embed, final_proj, norm, encode transformer layers, encode_norm.
    NO control flow — all modules are always present.

    Inputs:
        box_embed: (N, 1, C)    box embeddings from GeoProjections
        box_mask:  (1, N) bool  True=padded
        img_feat:  (HW, 1, C)  image features seq-first
        img_pos:   (HW, 1, C)  image position encoding seq-first

    Outputs:
        encoded: (N+1, 1, C)    +1 for CLS token (always last)
        mask:    (1, N+1) bool  CLS entry is always False (valid)
    """

    def __init__(self, geo):
        super().__init__()
        assert geo.cls_embed is not None, "cls_embed required"
        assert geo.final_proj is not None, "final_proj required"
        assert geo.encode is not None, "encode layers required"

        self.cls_embed = geo.cls_embed
        self.final_proj = geo.final_proj
        self.norm = geo.norm
        self.encode = geo.encode
        self.encode_norm = geo.encode_norm

    @torch.no_grad()
    def forward(
        self,
        box_embed: Tensor,  # (N, 1, C)
        box_mask: Tensor,  # (1, N) bool
        img_feat: Tensor,  # (HW, 1, C)
        img_pos: Tensor,  # (HW, 1, C)
    ):
        bs = box_embed.shape[1]

        # Append CLS token
        cls = self.cls_embed.weight.view(1, 1, -1).expand(1, bs, -1)
        cls_mask = torch.zeros(bs, 1, dtype=box_mask.dtype, device=box_mask.device)
        combined = torch.cat([box_embed, cls], dim=0)  # (N+1, 1, C)
        combined_mask = torch.cat([box_mask, cls_mask], dim=1)  # (1, N+1)

        # Final projection + norm
        combined = self.norm(self.final_proj(combined))

        # Cross-attention transformer (unrolled by trace)
        for lay in self.encode:
            combined = lay(
                tgt=combined,
                memory=img_feat,
                tgt_key_padding_mask=combined_mask,
                pos=img_pos,
            )
        combined = self.encode_norm(combined)

        return combined, combined_mask


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
        point_coords: Tensor,  # (B, N, 2)
        point_labels: Tensor,  # (B, N)
        has_box: Tensor,  # scalar indicator
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
            point_coords.shape[0],
            -1,
            self.prompt_encoder.image_embedding_size[0],
            self.prompt_encoder.image_embedding_size[1],
        )

        return sparse_embeddings, dense_embeddings


class Sam3SAM2MaskDecoderModel(nn.Module):
    """Wraps SAM2-style mask decoder from Sam3TrackerBase."""

    def __init__(self, model, multimask_output=True):
        super().__init__()
        self.mask_decoder = model.sam_mask_decoder
        self.prompt_encoder = model.sam_prompt_encoder
        self.multimask_output = multimask_output
        self.img_size = model.image_size

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: Tensor,  # (B, C, H, W)
        high_res_feats_0: Tensor,  # (B, C, H0, W0)
        high_res_feats_1: Tensor,  # (B, C, H1, W1)
        sparse_embeddings: Tensor,  # (B, N, C)
        dense_embeddings: Tensor,  # (B, C, H, W)
    ):
        # pe_layer is on PromptEncoder, not MaskDecoder
        image_pe = self.prompt_encoder.pe_layer(image_embeddings.shape[-2:]).unsqueeze(0)
        low_res_masks, iou_pred, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=[high_res_feats_0, high_res_feats_1],
        )

        # Upscale masks to image size
        high_res_masks = F.interpolate(low_res_masks, (self.img_size, self.img_size), mode="bilinear", align_corners=False)

        return low_res_masks, high_res_masks, iou_pred


class Sam3SAM1FeaturePrepModel(nn.Module):
    """
    Wraps conv_s0, conv_s1, and no_mem_embed for OpenVINO conversion.

    These are SAM1-task frozen weights used to prepare backbone features
    before the SAM2 prompt encoder / mask decoder.  Converting to OV IR
    eliminates the last torch weight ops from the inference pipeline.

    Inputs:
        fpn0: (B, C0, H0, W0)  — backbone FPN level 0
        fpn1: (B, C1, H1, W1)  — backbone FPN level 1

    Outputs:
        out0: conv_s0(fpn0)   — (B, C_out, H0, W0)
        out1: conv_s1(fpn1)   — (B, C_out, H1, W1)
        no_mem_embed          — (1, 1, C) constant
    """

    def __init__(self, conv_s0, conv_s1, no_mem_embed):
        super().__init__()
        self.conv_s0 = conv_s0
        self.conv_s1 = conv_s1
        self.register_buffer("no_mem_embed", no_mem_embed)

    @torch.no_grad()
    def forward(self, fpn0: Tensor, fpn1: Tensor):
        return self.conv_s0(fpn0), self.conv_s1(fpn1), self.no_mem_embed


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
    from pathlib import Path

    save_path = str(Path(save_path).resolve())
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


def convert_geometry_encoder_to_ov(model, save_dir: str, max_boxes: int = 10) -> None:
    """
    Convert the geometry encoder to OpenVINO IR.

    The geometry encoder (Sam3GeometryEncoderForOV) is converted to OV IR
    so the pipeline contains NO PyTorch modules with weights at inference.
    """
    import copy
    from pathlib import Path

    save_dir = Path(save_dir)
    ov_geo_path = save_dir / "geometry_encoder.xml"

    if ov_geo_path.exists():
        print(f"  [skip] Geometry Encoder already exists at {ov_geo_path}")
        return ov.Core().read_model(str(ov_geo_path))

    geo_encoder = copy.deepcopy(model.geometry_encoder).cpu()
    geo_wrapper = Sam3GeometryEncoderForOV(geo_encoder)
    geo_wrapper.eval()

    # Dummy inputs: (max_boxes, 1, 4), (1, max_boxes), (max_boxes, 1), (HW, 1, C), (HW, 1, C)
    C = 256
    HW = 72 * 72  # 5184
    dummy_box_emb = torch.zeros(max_boxes, 1, 4)
    dummy_box_mask = torch.ones(1, max_boxes, dtype=torch.bool)
    dummy_box_labels = torch.ones(max_boxes, 1, dtype=torch.long)
    dummy_img_feat = torch.randn(HW, 1, C)
    dummy_img_pos = torch.randn(HW, 1, C)

    ov_model = convert_and_save_model(
        wrapper_model=geo_wrapper,
        example_input=(dummy_box_emb, dummy_box_mask, dummy_box_labels, dummy_img_feat, dummy_img_pos),
        save_path=str(ov_geo_path),
        model_name="Geometry Encoder",
    )
    return ov_model


def convert_sam1_feature_prep_to_ov(model, save_dir: str):
    """
    Convert SAM1 feature-prep weights (conv_s0, conv_s1, no_mem_embed)
    to a single OpenVINO IR model.  Also saves scalar/list config as JSON.

    Returns the OV model, or None if the model has no SAM1 predictor.
    """
    import copy, json
    from pathlib import Path

    predictor = getattr(model, "inst_interactive_predictor", None)
    if predictor is None:
        print("  [skip] No SAM1 predictor found — skipping")
        return None

    save_dir = Path(save_dir)
    ov_path = save_dir / "sam1_feature_prep.xml"
    cfg_path = save_dir / "sam1_config.json"

    # Save scalar/list config (no weights)
    tracker_model = predictor.model
    config = {
        "bb_feat_sizes": predictor._bb_feat_sizes,
        "mask_threshold": float(predictor.mask_threshold),
        "image_size": int(tracker_model.image_size),
        "num_feature_levels": int(tracker_model.num_feature_levels),
    }
    with open(cfg_path, "w") as f:
        json.dump(config, f)
    print(f"  Saved SAM1 config to {cfg_path}")

    if ov_path.exists():
        print(f"  [skip] SAM1 Feature Prep already exists at {ov_path}")
        return ov.Core().read_model(str(ov_path))

    # Build wrapper from original model's conv layers
    wrapper = Sam3SAM1FeaturePrepModel(
        conv_s0=copy.deepcopy(tracker_model.sam_mask_decoder.conv_s0).cpu(),
        conv_s1=copy.deepcopy(tracker_model.sam_mask_decoder.conv_s1).cpu(),
        no_mem_embed=tracker_model.no_mem_embed.detach().cpu().clone(),
    )
    wrapper.eval()

    # Dummy inputs — spatial sizes from bb_feat_sizes (reversed: highest-res first)
    bb = predictor._bb_feat_sizes  # e.g. [(72,72), (36,36), (18,18)] or similar
    # conv_s0 is applied to fpn[0] (highest res), conv_s1 to fpn[1]
    c0_in = tracker_model.sam_mask_decoder.conv_s0.in_channels
    c1_in = tracker_model.sam_mask_decoder.conv_s1.in_channels
    h0, w0 = bb[0]
    h1, w1 = bb[1]
    dummy_fpn0 = torch.randn(1, c0_in, h0, w0)
    dummy_fpn1 = torch.randn(1, c1_in, h1, w1)

    ov_model = convert_and_save_model(
        wrapper_model=wrapper,
        example_input=(dummy_fpn0, dummy_fpn1),
        save_path=str(ov_path),
        model_name="SAM1 Feature Prep",
    )
    return ov_model


def load_sam1_config(path: str) -> dict:
    """Load SAM1 scalar/list config from JSON (no weights — those are in OV IR)."""
    import json

    with open(path, "r") as f:
        config = json.load(f)
    # Restore list-of-tuples structure
    config["bb_feat_sizes"] = [tuple(x) for x in config["bb_feat_sizes"]]
    return config


def save_geo_encoder_config(geo_encoder, save_dir: str):
    """
    Save geometry encoder constants needed by the split OV pipeline.

    Saves geo_config.json containing:
      - scalar config: roi_size, d_model, pos_enc params, flags
      - img_pre_norm weight/bias as JSON lists

    These are used by OVSam3Processor Python glue to run roi_align, sinusoidal
    pos encoding, and img_pre_norm without any PyTorch nn.Module.
    """
    import json
    from pathlib import Path

    save_dir = Path(save_dir)
    geo = geo_encoder

    config = {
        "roi_size": int(geo.roi_size),
        "d_model": int(geo.d_model),
        "has_direct": geo.boxes_direct_project is not None,
        "has_pool": geo.boxes_pool_project is not None,
        "has_pos_enc": geo.boxes_pos_enc_project is not None,
        "pos_enc_scale": float(geo.pos_enc.scale),
        "pos_enc_temperature": int(geo.pos_enc.temperature),
        "pos_enc_num_feats": int(geo.pos_enc.num_pos_feats),
    }

    if geo.boxes_pool_project is not None and hasattr(geo.img_pre_norm, "weight"):
        config["img_pre_norm_weight"] = geo.img_pre_norm.weight.detach().cpu().tolist()
        config["img_pre_norm_bias"] = geo.img_pre_norm.bias.detach().cpu().tolist()

    cfg_path = save_dir / "geo_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f)
    print(f"  Saved geo config to {cfg_path}")

    return config


def load_geo_encoder_config(save_dir: str):
    """
    Load geometry encoder config and constants from geo_config.json.
    Returns (config_dict, constants_dict).
    constants_dict contains torch tensors for img_pre_norm_weight/bias if present.
    """
    import json
    from pathlib import Path

    save_dir = Path(save_dir)

    with open(save_dir, "r") as f:
        config = json.load(f)

    constants = {}
    if "img_pre_norm_weight" in config:
        constants["img_pre_norm_weight"] = torch.tensor(config.pop("img_pre_norm_weight"), dtype=torch.float32)
        constants["img_pre_norm_bias"] = torch.tensor(config.pop("img_pre_norm_bias"), dtype=torch.float32)

    return config, constants


def compute_sinusoidal_pos_enc(cx, cy, w, h, scale, temperature, num_pos_feats):
    """
    Compute sinusoidal position encoding for boxes — pure math, no learned weights.
    Replicates PositionEmbeddingSine.encode_boxes / _encode_xy.

    Args:
        cx, cy, w, h: (N,) tensors of normalized box coordinates
        scale, temperature, num_pos_feats: pos_enc hyperparameters

    Returns:
        pos: (N, 2*num_pos_feats + 2) = (N, d_model+2) position features
    """
    x_embed = cx * scale
    y_embed = cy * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=cx.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)

    return torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)


def _get_dummy_prompt(device="cpu"):
    """Standalone version of Sam3Image._get_dummy_prompt() — no model needed."""
    import torch
    from sam3.model.geometry_encoders import Prompt

    return Prompt(
        box_embeddings=torch.zeros(0, 1, 4, device=device),
        box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
    )


def _get_img_feats(backbone_out, img_ids, num_feature_levels=1):
    """Standalone version of Sam3Image._get_img_feats() — no model needed.
    num_feature_levels=1 matches the default Sam3Image configuration."""
    import torch

    vis_feats = backbone_out["backbone_fpn"][-num_feature_levels:]
    vis_pos_enc = backbone_out["vision_pos_enc"][-num_feature_levels:]
    vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
    img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
    img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
    return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes


class OVSam3Processor:
    """
    Drop-in replacement for Sam3Processor using OpenVINO compiled models.
    Mirrors the Sam3Processor API: set_image, set_text_prompt, add_geometric_prompt.
    """

    def __init__(
        self,
        ov_image_encoder,
        ov_text_encoder,
        ov_transformer_encoder,
        ov_decoder,  # transformer decoder (separate from scoring)
        ov_scoring,  # scoring + bbox_embed + decoder_norm
        ov_seg_head,
        tokenizer,
        ov_prompt_encoder=None,  # OV compiled SAM2 prompt encoder (for SAM1 task)
        ov_mask_decoder=None,  # OV compiled SAM2 mask decoder (for SAM1 task)
        ov_geometry_encoder=None,  # OV compiled geometry encoder (legacy single model)
        ov_geo_projections=None,  # OV compiled geo projections (split model)
        ov_geo_cross_attn=None,  # OV compiled geo cross-attention (split model)
        geo_config_path=None,  # path to directory with geo_config.json + geo_constants.npz
        ov_sam1_feature_prep=None,  # OV compiled SAM1 feature prep (conv_s0/s1 + no_mem_embed)
        sam1_config_path=None,  # path to sam1_config.json (scalar config only)
        resolution=1008,
        confidence_threshold=0.5,
        max_boxes=10,
    ):
        self.ov_image_encoder = ov_image_encoder
        self.ov_text_encoder = ov_text_encoder
        self.ov_transformer_encoder = ov_transformer_encoder
        self.ov_decoder = ov_decoder
        self.ov_scoring = ov_scoring
        self.ov_seg_head = ov_seg_head
        self.ov_prompt_encoder = ov_prompt_encoder
        self.ov_mask_decoder = ov_mask_decoder
        self.ov_geometry_encoder = ov_geometry_encoder
        self.ov_geo_projections = ov_geo_projections
        self.ov_geo_cross_attn = ov_geo_cross_attn
        self.ov_sam1_feature_prep = ov_sam1_feature_prep
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold
        self.max_boxes = max_boxes

        # Split geometry encoder config (Python glue for non-weighted ops)
        self._geo_split_ready = False
        self._geo_config = None
        self._geo_pre_norm_weight = None
        self._geo_pre_norm_bias = None
        if geo_config_path is not None and ov_geo_projections is not None:
            self._geo_config, geo_constants = load_geo_encoder_config(geo_config_path)
            self._geo_pre_norm_weight = geo_constants.get("img_pre_norm_weight")
            self._geo_pre_norm_bias = geo_constants.get("img_pre_norm_bias")
            self._geo_split_ready = True

        # Load SAM1 config from JSON
        sam1_config = None
        if sam1_config_path is not None:
            sam1_config = load_sam1_config(sam1_config_path)

        # SAM1 task config (scalar/list values — weights are in ov_sam1_feature_prep)
        self._sam1_ready = False
        self._bb_feat_sizes = None
        self._mask_threshold = 0.0
        self._sam1_image_size = resolution
        self._sam1_num_feature_levels = 3

        if sam1_config is not None:
            self._bb_feat_sizes = sam1_config.get("bb_feat_sizes")
            self._mask_threshold = sam1_config.get("mask_threshold", 0.0)
            self._sam1_image_size = sam1_config.get("image_size", resolution)
            self._sam1_num_feature_levels = sam1_config.get("num_feature_levels", 3)
            self._sam1_ready = True

        from torchvision.transforms import v2

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

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

        # Parse outputs: sam3_fpn[0..2], sam3_pos[0..2], [sam2_fpn[0..2], sam2_pos[0..2]]
        n_outputs = len(ov_result)
        has_sam2 = n_outputs > 6  # 12 outputs if sam2 neck exists
        n_levels = 3  # always 3 levels after scalp=1

        backbone_fpn = [torch.from_numpy(np.array(ov_result[i])) for i in range(n_levels)]
        vision_pos_enc = [torch.from_numpy(np.array(ov_result[n_levels + i])) for i in range(n_levels)]

        state["backbone_out"] = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": vision_pos_enc,
            "vision_features": backbone_fpn[-1],
        }

        if has_sam2:
            sam2_offset = 2 * n_levels
            sam2_fpn = [torch.from_numpy(np.array(ov_result[sam2_offset + i])) for i in range(n_levels)]
            sam2_pos = [torch.from_numpy(np.array(ov_result[sam2_offset + n_levels + i])) for i in range(n_levels)]
            state["backbone_out"]["sam2_backbone_out"] = {
                "backbone_fpn": sam2_fpn,
                "vision_pos_enc": sam2_pos,
                "vision_features": sam2_fpn[-1],
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
        text_memory_resized = torch.from_numpy(np.array(ov_result[0]))
        text_attention_mask = torch.from_numpy(np.array(ov_result[1]))

        state["backbone_out"]["language_features"] = text_memory_resized
        state["backbone_out"]["language_mask"] = text_attention_mask
        state["backbone_out"]["language_embeds"] = text_memory_resized  # simplified

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = _get_dummy_prompt()

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
            state["backbone_out"]["language_features"] = torch.from_numpy(np.array(ov_result[0]))
            state["backbone_out"]["language_mask"] = torch.from_numpy(np.array(ov_result[1]))
            state["backbone_out"]["language_embeds"] = torch.from_numpy(np.array(ov_result[0]))

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = _get_dummy_prompt()

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
        """Run the full grounding pipeline using OV models (no PyTorch model weights at inference)."""
        backbone_out = state["backbone_out"]
        geometric_prompt = state["geometric_prompt"]

        # Step 1: Prepare image features
        txt_ids = self.find_stage.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]

        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = _get_img_feats(backbone_out, self.find_stage.img_ids)

        # Step 2: Prepare box inputs (pad to max_boxes for OV geometry encoder)
        geo = geometric_prompt.clone()
        raw_boxes = geo.box_embeddings  # (N_boxes, B, 4) or zeros(0,1,4)
        raw_mask = geo.box_mask  # (B, N_boxes) or zeros(1,0)
        raw_labels = geo.box_labels  # (N_boxes, B) or zeros(N,B)

        n_actual = raw_boxes.shape[0]
        pad = self.max_boxes - n_actual
        if pad < 0:
            raise ValueError(f"Got {n_actual} boxes but max_boxes={self.max_boxes}")

        if pad > 0:
            zero_boxes = torch.zeros(pad, 1, 4)
            pad_mask = torch.ones(1, pad, dtype=torch.bool)  # True = padded
            pad_labels = torch.ones(pad, 1, dtype=torch.long)  # label 1 (ignored by mask)
            box_embeddings = torch.cat([raw_boxes, zero_boxes], dim=0)
            box_mask = torch.cat([raw_mask, pad_mask], dim=1)
            box_labels_t = torch.cat([raw_labels, pad_labels], dim=0)
        else:
            box_embeddings = raw_boxes
            box_mask = raw_mask
            box_labels_t = raw_labels

        # Step 3: Run geometry encoder
        if self._geo_split_ready:
            # Split OV path — weighted ops in OV, non-weighted ops in Python
            geo_feats, geo_masks = self._run_geo_split(
                box_embeddings,
                box_mask,
                box_labels_t,
                img_feats[0],
                img_pos_embeds[0],
            )
        elif self.ov_geometry_encoder is not None:
            # Legacy single OV model path
            geo_result = self.ov_geometry_encoder(
                [
                    box_embeddings.numpy(),
                    box_mask.numpy(),
                    box_labels_t.numpy(),
                    img_feats[0].numpy(),
                    img_pos_embeds[0].numpy(),
                ]
            )
            geo_feats = torch.from_numpy(np.array(geo_result[0]))
            geo_masks = torch.from_numpy(np.array(geo_result[1]))
        else:
            raise RuntimeError(
                "No geometry encoder configured. Pass ov_geo_projections + ov_geo_cross_attn + geo_config_path "
                "(split pipeline) or ov_geometry_encoder (legacy single model) to OVSam3Processor."
            )
        geo_masks = geo_masks.bool()

        # Step 4: Assemble combined prompt for transformer encoder
        prompt = torch.cat([txt_feats, geo_feats], dim=0)  # (S_text + max_boxes+1, B, C)
        prompt_mask = torch.cat([txt_masks, geo_masks], dim=1)  # (B, S_text + max_boxes+1)

        # Step 5: Run OV transformer encoder (positional inputs)
        enc_result = self.ov_transformer_encoder(
            [
                img_feats[0].numpy(),  # 0: img_feat
                img_pos_embeds[0].numpy(),  # 1: img_pos
                prompt.numpy(),  # 2: prompt (OV name may be auto-generated)
                prompt_mask.numpy(),  # 3: prompt_mask
            ]
        )

        memory = torch.from_numpy(np.array(enc_result[0]))
        pos_embed = torch.from_numpy(np.array(enc_result[1]))
        level_start_index = torch.from_numpy(np.array(enc_result[3]))
        spatial_shapes = torch.from_numpy(np.array(enc_result[4]))
        valid_ratios = torch.from_numpy(np.array(enc_result[5]))

        # Step 6a: Run OV transformer decoder
        dec_result = self.ov_decoder(
            [
                memory.numpy(),  # 0: memory
                pos_embed.numpy(),  # 1: pos_embed
                torch.zeros(1).numpy(),  # 2: memory_mask
                level_start_index.numpy(),  # 3: level_start_index
                spatial_shapes.numpy(),  # 4: spatial_shapes
                valid_ratios.numpy(),  # 5: valid_ratios
                prompt.numpy(),  # 6: prompt
                prompt_mask.numpy(),  # 7: prompt_mask
            ]
        )

        hs = torch.from_numpy(np.array(dec_result[0]))  # (num_layers, nq, bs, C) seq-first
        reference_boxes = torch.from_numpy(np.array(dec_result[1]))  # (num_layers+1, nq, bs, 4) seq-first
        presence_logits = torch.from_numpy(np.array(dec_result[2]))

        # Transpose to batch-first for scoring
        hs_bf = hs.transpose(1, 2)  # (num_layers, bs, nq, C)
        reference_boxes_bf = reference_boxes.transpose(1, 2)  # (num_layers+1, bs, nq, 4)

        # Step 6b: Run OV scoring + box refinement
        scoring_result = self.ov_scoring(
            [
                hs_bf.numpy(),  # 0: hs (batch-first)
                reference_boxes_bf.numpy(),  # 1: reference_boxes (batch-first)
                prompt.numpy(),  # 2: prompt
                prompt_mask.numpy(),  # 3: prompt_mask
            ]
        )

        pred_logits = torch.from_numpy(np.array(scoring_result[0]))
        pred_boxes = torch.from_numpy(np.array(scoring_result[1]))  # (bs, nq, 4) cxcywh [0,1]

        # Step 7: Run OV segmentation head (positional inputs)
        fpn = backbone_out["backbone_fpn"]
        seg_result = self.ov_seg_head(
            [
                fpn[0].numpy(),  # 0: fpn0
                fpn[1].numpy(),  # 1: fpn1
                fpn[2].numpy(),  # 2: fpn2
                hs_bf[-1].numpy(),  # 3: obj_queries (OV name may be auto-generated)
                memory.numpy(),  # 4: encoder_hidden_states
                prompt.numpy(),  # 5: prompt
                prompt_mask.numpy(),  # 6: prompt_mask
            ]
        )
        pred_masks = torch.from_numpy(np.array(seg_result[0]))

        # Step 8: Post-process
        # presence_logits from OV decoder is (num_layers, 1, bs) — seq-first, NOT transposed.
        # Original code (sam3_image.py:286) transposes to (num_layers, bs, 1) before _update_out[-1].
        # So: [-1] gives (1, bs); .T → (bs, 1); .unsqueeze(1) → (bs, 1, 1) for correct broadcasting.
        out_logits = pred_logits[-1]  # last layer (bs, nq, 1)
        presence_score = (
            presence_logits[-1].T.unsqueeze(1).sigmoid() if presence_logits.numel() > 1 else torch.ones_like(out_logits)  # (1,bs) → (bs,1) → (bs,1,1)
        )
        out_probs = (out_logits.sigmoid() * presence_score).squeeze(-1)  # (bs, nq)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = pred_masks[keep]
        out_bbox = pred_boxes[keep]  # already cxcywh [0,1]

        from sam3.model import box_ops

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        boxes = boxes * scale_fct[None, :]

        from sam3.model.data_misc import interpolate

        out_masks = interpolate(out_masks.unsqueeze(1), (img_h, img_w), mode="bilinear", align_corners=False).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    def _run_geo_split(self, box_embeddings, box_mask, box_labels_t, img_feat, img_pos):
        """
        Run split geometry encoder: Python glue for non-weighted ops +
        two OV models for weighted ops.

        Non-weighted ops in Python:
          - sinusoidal position encoding (compute_sinusoidal_pos_enc)
          - img_pre_norm via F.layer_norm with extracted weights
          - roi_align via torchvision.ops (no learned weights)
          - box_cxcywh_to_xyxy (pure math)

        OV models:
          - ov_geo_projections: Linear projections + Embedding (no control flow)
          - ov_geo_cross_attn: CLS + final_proj + cross-attention layers (no control flow)
        """
        import torchvision.ops

        cfg = self._geo_config
        n_boxes = box_embeddings.shape[0]
        bs = box_embeddings.shape[1]
        d_model = cfg["d_model"]

        # --- Python: sinusoidal position encoding (no weights) ---
        cx = box_embeddings[:, :, 0].flatten()  # (N*B,)
        cy = box_embeddings[:, :, 1].flatten()
        w = box_embeddings[:, :, 2].flatten()
        h = box_embeddings[:, :, 3].flatten()
        pos_enc_feat = compute_sinusoidal_pos_enc(
            cx,
            cy,
            w,
            h,
            scale=cfg["pos_enc_scale"],
            temperature=cfg["pos_enc_temperature"],
            num_pos_feats=cfg["pos_enc_num_feats"],
        )  # (N*B, d_model+2)
        pos_features = pos_enc_feat.view(n_boxes, bs, -1)  # (N, 1, d_model+2)

        # --- Python: img_pre_norm + roi_align (no learned weights in OV) ---
        feat_h = feat_w = int(img_feat.shape[0] ** 0.5)  # 72
        normed_img = F.layer_norm(
            img_feat,
            [d_model],
            self._geo_pre_norm_weight,
            self._geo_pre_norm_bias,
        )  # (HW, 1, C)
        img_nchw = normed_img.permute(1, 2, 0).view(bs, -1, feat_h, feat_w)  # (1, C, H, W)

        # box_cxcywh_to_xyxy (pure math)
        bcx, bcy, bw, bh = box_embeddings.unbind(-1)
        boxes_xyxy = torch.stack(
            [
                bcx - 0.5 * bw,
                bcy - 0.5 * bh,
                bcx + 0.5 * bw,
                bcy + 0.5 * bh,
            ],
            dim=-1,
        )  # (N, 1, 4)
        scale = torch.tensor([feat_w, feat_h, feat_w, feat_h], dtype=boxes_xyxy.dtype).view(1, 1, 4)
        boxes_xyxy_scaled = boxes_xyxy * scale

        # Tensor-format roi_align: (K, 5) = [batch_idx, x1, y1, x2, y2]
        boxes_2d = boxes_xyxy_scaled[:, 0, :]  # (N, 4)
        batch_idx = torch.zeros(n_boxes, 1, dtype=boxes_xyxy.dtype)
        boxes_with_batch = torch.cat([batch_idx, boxes_2d], dim=1)  # (N, 5)
        roi_size = cfg["roi_size"]
        roi_features = torchvision.ops.roi_align(
            img_nchw,
            boxes_with_batch,
            roi_size,
        )  # (N, C, roi_size, roi_size)

        # --- OV: geo_projections (weighted ops only, no control flow) ---
        proj_result = self.ov_geo_projections(
            [
                box_embeddings.numpy(),
                pos_features.numpy(),
                roi_features.detach().numpy(),
                box_labels_t.numpy(),
            ]
        )
        box_embed = torch.from_numpy(np.array(proj_result[0]))  # (N, 1, C)

        # --- OV: geo_cross_attn (CLS + cross-attention, no control flow) ---
        cross_result = self.ov_geo_cross_attn(
            [
                box_embed.numpy(),
                box_mask.numpy(),
                img_feat.numpy(),
                img_pos.numpy(),
            ]
        )
        geo_feats = torch.from_numpy(np.array(cross_result[0]))  # (N+1, 1, C)
        geo_masks = torch.from_numpy(np.array(cross_result[1]))  # (1, N+1)

        return geo_feats, geo_masks

    @torch.inference_mode()
    def predict_inst(self, state, **kwargs):
        """
        SAM1-style point/box prompting using pre-computed sam2 features.
        All inference uses OV models + extracted frozen weights (no PyTorch nn.Module).
        """
        if "backbone_out" not in state or "sam2_backbone_out" not in state["backbone_out"]:
            raise ValueError("No sam2 features. set_image must be called with sam2-enabled encoder.")

        if not self._sam1_ready:
            raise RuntimeError("SAM1 task requires config. Pass sam1_config_path= to OVSam3Processor.")
        if self.ov_sam1_feature_prep is None:
            raise RuntimeError("SAM1 task requires ov_sam1_feature_prep. Compile sam1_feature_prep.xml and pass it to OVSam3Processor.")
        if self.ov_prompt_encoder is None or self.ov_mask_decoder is None:
            raise RuntimeError("SAM1 task requires ov_prompt_encoder and ov_mask_decoder. " "Compile and pass them to OVSam3Processor.")

        # Prepare features using OV sam1_feature_prep model (conv_s0/s1 + no_mem_embed)
        backbone_out = state["backbone_out"]["sam2_backbone_out"]
        backbone_out = backbone_out.copy()
        backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])

        # Run OV feature prep: conv_s0(fpn0), conv_s1(fpn1), no_mem_embed
        prep_result = self.ov_sam1_feature_prep([backbone_out["backbone_fpn"][0].numpy(), backbone_out["backbone_fpn"][1].numpy()])
        backbone_out["backbone_fpn"][0] = torch.from_numpy(np.array(prep_result[0]))
        backbone_out["backbone_fpn"][1] = torch.from_numpy(np.array(prep_result[1]))
        no_mem_embed = torch.from_numpy(np.array(prep_result[2]))

        # _prepare_backbone_features inline (no model dependency)
        feature_maps = backbone_out["backbone_fpn"][-self._sam1_num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self._sam1_num_feature_levels :]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]

        # Add no_mem_embed (from OV model output)
        vision_feats[-1] = vision_feats[-1] + no_mem_embed

        # Reshape to NCHW
        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])][::-1]

        image_embed = feats[-1]  # (1, C, H, W) lowest res
        high_res_feats = feats[:-1]  # list of higher res features

        # Prepare point/box prompts
        point_coords = kwargs.get("point_coords")
        point_labels = kwargs.get("point_labels")
        box = kwargs.get("box")
        multimask_output = kwargs.get("multimask_output", True)
        normalize_coords = kwargs.get("normalize_coords", True)

        orig_h = state["original_height"]
        orig_w = state["original_width"]

        if point_coords is not None:
            point_coords = torch.as_tensor(point_coords, dtype=torch.float32)
            point_labels = torch.as_tensor(point_labels, dtype=torch.int32)
            if normalize_coords:
                point_coords = point_coords.clone()
                point_coords[..., 0] = point_coords[..., 0] / orig_w * self._sam1_image_size
                point_coords[..., 1] = point_coords[..., 1] / orig_h * self._sam1_image_size

        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float32)
            if normalize_coords:
                box = box.clone()
                box[..., 0] = box[..., 0] / orig_w * self._sam1_image_size
                box[..., 1] = box[..., 1] / orig_h * self._sam1_image_size
                box[..., 2] = box[..., 2] / orig_w * self._sam1_image_size
                box[..., 3] = box[..., 3] / orig_h * self._sam1_image_size

        # Combine point and box prompts
        if box is not None:
            box_corners = box.reshape(-1, 2, 2)
            box_labels_t = torch.tensor([2, 3], dtype=torch.int32)
            if point_coords is not None:
                point_coords = torch.cat(
                    [
                        point_coords if point_coords.dim() == 2 else point_coords.squeeze(0),
                        box_corners,
                    ],
                    dim=0,
                )
                point_labels = torch.cat([point_labels.flatten(), box_labels_t], dim=0)
            else:
                point_coords = box_corners
                point_labels = box_labels_t

        if point_coords.dim() == 2:
            point_coords = point_coords.unsqueeze(0)
        if point_labels.dim() == 1:
            point_labels = point_labels.unsqueeze(0)

        # Add padding point
        padding_point = torch.zeros((point_coords.shape[0], 1, 2))
        padding_label = -torch.ones((point_labels.shape[0], 1), dtype=torch.int32)
        concat_coords = torch.cat([point_coords, padding_point], dim=1)
        concat_labels = torch.cat([point_labels, padding_label], dim=1)

        # Run OV prompt encoder
        has_box = torch.tensor(1 if box is not None else 0)
        enc_result = self.ov_prompt_encoder(
            [
                concat_coords.numpy(),
                concat_labels.numpy(),
                has_box.numpy(),
            ]
        )
        sparse_embeddings = torch.from_numpy(np.array(enc_result[0]))
        dense_embeddings = torch.from_numpy(np.array(enc_result[1]))

        # Run OV mask decoder
        dec_result = self.ov_mask_decoder(
            [
                image_embed.numpy(),
                high_res_feats[0].numpy(),
                high_res_feats[1].numpy(),
                sparse_embeddings.numpy(),
                dense_embeddings.numpy(),
            ]
        )

        low_res_masks = torch.from_numpy(np.array(dec_result[0]))
        high_res_masks = torch.from_numpy(np.array(dec_result[1]))
        iou_pred = torch.from_numpy(np.array(dec_result[2]))

        # Post-process to original image size
        masks = F.interpolate(
            high_res_masks,
            (orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        masks = (masks > self._mask_threshold).squeeze(0).numpy()
        scores = iou_pred.squeeze(0).numpy()
        logits = low_res_masks.squeeze(0).numpy()

        # OV model was exported with multimask_output=True, so it always
        # returns multiple masks.  When the caller requests single-mask
        # output, select the best one by IoU score.
        if not multimask_output and masks.shape[0] > 1:
            best = np.argmax(scores.flatten())
            masks = masks[best : best + 1]
            scores = scores.flatten()[best : best + 1]
            logits = logits[best : best + 1]

        return masks, scores, logits


class OVSam3InteractiveImagePredictor:
    """
    Drop-in replacement for SAM3InteractiveImagePredictor using OV models.
    Used for SAM1-style point/box prompting.
    All inference uses OV models + extracted frozen weights (no PyTorch nn.Module).
    """

    def __init__(
        self,
        original_predictor,  # SAM3InteractiveImagePredictor
        ov_image_encoder,
        ov_prompt_encoder,
        ov_mask_decoder,
        ov_sam1_feature_prep,  # OV compiled SAM1 feature prep model
        image_size=1008,
    ):
        self.ov_image_encoder = ov_image_encoder
        self.ov_prompt_encoder = ov_prompt_encoder
        self.ov_mask_decoder = ov_mask_decoder
        self.ov_sam1_feature_prep = ov_sam1_feature_prep
        self.image_size = image_size
        self._transforms = original_predictor._transforms
        self._features = None
        self._orig_hw = None
        self._bb_feat_sizes = original_predictor._bb_feat_sizes
        self.mask_threshold = original_predictor.mask_threshold
        self._num_feature_levels = original_predictor.model.num_feature_levels

    @torch.no_grad()
    def set_image(self, image):
        """Encode image for point/box prompting using shared OV image encoder."""
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
        has_sam2 = n_outputs > 6
        n_levels = 3

        if has_sam2:
            # Use sam2 features (outputs 6..11)
            sam2_offset = 2 * n_levels
            sam2_fpn = [torch.from_numpy(np.array(ov_result[sam2_offset + i])) for i in range(n_levels)]
            sam2_pos = [torch.from_numpy(np.array(ov_result[sam2_offset + n_levels + i])) for i in range(n_levels)]
        else:
            # Fallback: use sam3 features
            sam2_fpn = [torch.from_numpy(np.array(ov_result[i])) for i in range(n_levels)]
            sam2_pos = [torch.from_numpy(np.array(ov_result[n_levels + i])) for i in range(n_levels)]

        # Run OV feature prep: conv_s0(fpn0), conv_s1(fpn1), no_mem_embed
        prep_result = self.ov_sam1_feature_prep([sam2_fpn[0].numpy(), sam2_fpn[1].numpy()])
        sam2_fpn[0] = torch.from_numpy(np.array(prep_result[0]))
        sam2_fpn[1] = torch.from_numpy(np.array(prep_result[1]))
        no_mem_embed = torch.from_numpy(np.array(prep_result[2]))

        # _prepare_backbone_features inline (no model dependency)
        backbone_out = {
            "backbone_fpn": sam2_fpn,
            "vision_pos_enc": sam2_pos,
            "vision_features": sam2_fpn[-1],
        }
        feature_maps = backbone_out["backbone_fpn"][-self._num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self._num_feature_levels :]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]

        # Add no_mem_embed (from OV model output)
        vision_feats[-1] = vision_feats[-1] + no_mem_embed

        # Reshape to NCHW (same as predict_inst)
        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])][::-1]

        self._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
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
                point_coords = (
                    torch.cat([point_coords, box_corners], dim=0) if point_coords.dim() == 2 else torch.cat([point_coords.squeeze(0), box_corners], dim=0)
                )
                point_labels = torch.cat([point_labels.flatten(), box_labels], dim=0)
            else:
                point_coords = box_corners
                point_labels = box_labels

        if point_coords.dim() == 2:
            point_coords = point_coords.unsqueeze(0)
        if point_labels.dim() == 1:
            point_labels = point_labels.unsqueeze(0)

        # Add padding point
        padding_point = torch.zeros((point_coords.shape[0], 1, 2))
        padding_label = -torch.ones((point_labels.shape[0], 1), dtype=torch.int32)
        concat_coords = torch.cat([point_coords, padding_point], dim=1)
        concat_labels = torch.cat([point_labels, padding_label], dim=1)

        # Run OV prompt encoder (positional inputs)
        has_box = torch.tensor(1 if box is not None else 0)
        enc_result = self.ov_prompt_encoder(
            [
                concat_coords.numpy(),  # 0: point_coords
                concat_labels.numpy(),  # 1: point_labels
                has_box.numpy(),  # 2: has_box
            ]
        )
        sparse_embeddings = torch.from_numpy(np.array(enc_result[0]))
        dense_embeddings = torch.from_numpy(np.array(enc_result[1]))

        # Run OV mask decoder (positional inputs)
        image_embed = self._features["image_embed"]
        high_res_feats = self._features["high_res_feats"]

        dec_result = self.ov_mask_decoder(
            [
                image_embed.numpy(),  # 0: image_embeddings
                high_res_feats[0].numpy(),  # 1: high_res_feats_0
                high_res_feats[1].numpy(),  # 2: high_res_feats_1
                sparse_embeddings,  # 3: sparse_embeddings
                dense_embeddings,  # 4: dense_embeddings
            ]
        )

        low_res_masks = torch.from_numpy(np.array(dec_result[0]))
        high_res_masks = torch.from_numpy(np.array(dec_result[1]))
        iou_pred = torch.from_numpy(np.array(dec_result[2]))

        # Post-process to original image size
        orig_h, orig_w = self._orig_hw[0]
        masks = F.interpolate(
            high_res_masks,
            (orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
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
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="green", facecolor="none")
            plt.gca().add_patch(rect)
            if scores is not None:
                score = scores[i] if isinstance(scores[i], float) else scores[i].item()
                plt.text(x0, y0 - 5, f"{score:.2f}", color="green", fontsize=10)

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

    # Combine all masks into single 2D image for visualization
    pt_combined = pt_binary.max(axis=tuple(range(pt_binary.ndim - 2))) if pt_binary.ndim > 2 else pt_binary
    ov_combined = ov_binary.max(axis=tuple(range(ov_binary.ndim - 2))) if ov_binary.ndim > 2 else ov_binary

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pt_combined, cmap="gray")
    axes[0].set_title("PyTorch")
    axes[0].axis("off")

    axes[1].imshow(ov_combined, cmap="gray")
    axes[1].set_title("OpenVINO")
    axes[1].axis("off")

    diff = np.abs(pt_combined - ov_combined)
    axes[2].imshow(diff, cmap="hot")
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
