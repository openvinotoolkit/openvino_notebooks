"""InternVLA-N1 System 1 modules used for OpenVINO export."""

from __future__ import annotations

import math
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from diffusers.models.attention import LuminaFeedForward
from diffusers.models.attention_processor import Attention, LuminaAttnProcessor2_0
from diffusers.models.embeddings import (
    LuminaCombinedTimestepCaptionEmbedding,
    LuminaPatchEmbed,
    PixArtAlphaTextProjection,
)
from diffusers.models.normalization import (
    LuminaLayerNormContinuous,
    LuminaRMSNormZero,
    RMSNorm,
)
from torch import nn

from .checkpoint import load_module_prefix


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        exponent = -torch.arange(embedding_dim // 2, dtype=torch.float32)
        exponent *= math.log(10000.0) / (embedding_dim // 2)
        self.register_buffer("frequencies", exponent.exp(), persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.to(torch.float32).unsqueeze(-1) * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class MemoryEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=6,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.memory_pos = nn.Parameter(torch.empty(512, 384))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.encoder(memory + self.memory_pos[: memory.shape[1]].unsqueeze(0))


class QFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(32, 768))
        self.query_pos = nn.Parameter(torch.empty(32, 768))
        layer = nn.TransformerDecoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=2048,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=3)
        self.visual_proj = nn.Linear(768, 768)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        batch_size = visual_features.shape[0]
        queries = (self.query_tokens + self.query_pos).unsqueeze(0)
        return self.decoder(queries.expand(batch_size, -1, -1), visual_features)


def create_rgb_backbone() -> nn.Module:
    return timm.create_model(
        "vit_small_patch14_dinov2",
        pretrained=False,
        img_size=518,
        dynamic_img_size=True,
        num_classes=0,
    )


class System1ContextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rgb_model = create_rgb_backbone()
        self.memory_encoder = MemoryEncoder()
        self.rgb_resampler = QFormer()
        self.cond_projector = nn.Sequential(
            nn.Linear(3584, 768),
            nn.GELU(approximate="tanh"),
            nn.Linear(768, 768),
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def _rgb_features(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.rgb_model.patch_embed(images).flatten(1, 2)
        tokens = torch.cat(
            (self.rgb_model.cls_token.expand(tokens.shape[0], -1, -1), tokens),
            dim=1,
        )
        patch_positions = self.rgb_model.pos_embed[:, 1:].float()
        grid_size = int(math.sqrt(patch_positions.shape[1]))
        patch_positions = F.interpolate(
            patch_positions.reshape(1, grid_size, grid_size, 384).permute(0, 3, 1, 2),
            scale_factor=(16.1 / 37, 16.1 / 37),
            mode="bicubic",
            antialias=False,
        )
        patch_positions = patch_positions.permute(0, 2, 3, 1).flatten(1, 2)
        positions = torch.cat((self.rgb_model.pos_embed[:, :1], patch_positions), dim=1)
        tokens = tokens + positions.to(tokens.dtype)
        tokens = self.rgb_model.blocks(tokens)
        return self.rgb_model.norm(tokens)[:, 1:]

    def forward(self, images: torch.Tensor, trajectory_latents: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count = images.shape[:2]
        normalized = (images - self.image_mean) / self.image_std
        features = self._rgb_features(normalized.flatten(0, 1))
        features = features.unflatten(0, (batch_size, frame_count))
        raw_memory = features.flatten(1, 2)
        encoded_memory = self.memory_encoder(raw_memory)
        memory_tokens = self.rgb_resampler(torch.cat((raw_memory, encoded_memory), dim=-1))
        return torch.cat((memory_tokens, self.cond_projector(trajectory_latents)), dim=1)


class NextDiTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(6))
        self.attn1 = Attention(
            query_dim=384,
            cross_attention_dim=None,
            dim_head=64,
            qk_norm="layer_norm_across_heads",
            heads=6,
            kv_heads=6,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )
        self.attn1.to_out = nn.Identity()
        self.attn2 = Attention(
            query_dim=384,
            cross_attention_dim=384,
            dim_head=64,
            qk_norm="layer_norm_across_heads",
            heads=6,
            kv_heads=6,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )
        self.feed_forward = LuminaFeedForward(
            dim=384,
            inner_dim=1536,
            multiple_of=256,
            ffn_dim_multiplier=2 / 3,
        )
        self.norm1 = LuminaRMSNormZero(
            embedding_dim=384,
            norm_eps=1e-5,
            norm_elementwise_affine=True,
        )
        self.ffn_norm1 = RMSNorm(384, eps=1e-5, elementwise_affine=True)
        self.norm2 = RMSNorm(384, eps=1e-5, elementwise_affine=True)
        self.ffn_norm2 = RMSNorm(384, eps=1e-5, elementwise_affine=True)
        self.norm1_context = RMSNorm(384, eps=1e-5, elementwise_affine=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized, attention_gate, mlp_scale, mlp_gate = self.norm1(hidden_states, time_embedding)
        self_attention = self.attn1(
            hidden_states=normalized,
            encoder_hidden_states=normalized,
            attention_mask=attention_mask,
            query_rotary_emb=None,
            key_rotary_emb=None,
        )
        cross_attention = self.attn2(
            hidden_states=normalized,
            encoder_hidden_states=self.norm1_context(encoder_hidden_states),
            attention_mask=encoder_mask,
            query_rotary_emb=None,
            key_rotary_emb=None,
        )
        cross_attention = cross_attention * self.gate.tanh().view(1, 1, 6, 1)
        mixed_attention = (self_attention + cross_attention).flatten(-2)
        hidden_states = self.attn2.to_out[0](mixed_attention)
        hidden_states = residual + attention_gate.unsqueeze(1).tanh() * self.norm2(hidden_states)
        mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + mlp_scale.unsqueeze(1)))
        return hidden_states + mlp_gate.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)


class NextDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.caption_projection = PixArtAlphaTextProjection(in_features=768, hidden_size=384)
        self.patch_embedder = LuminaPatchEmbed(patch_size=1, in_channels=384, embed_dim=384, bias=True)
        self.time_caption_embed = LuminaCombinedTimestepCaptionEmbedding(hidden_size=384, cross_attention_dim=384)
        self.layers = nn.ModuleList(NextDiTBlock() for _ in range(12))
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=384,
            conditioning_embedding_dim=384,
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=384,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.int32, device=hidden_states.device)
        encoder_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.int32,
            device=encoder_hidden_states.device,
        )
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        time_embedding = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_mask,
                time_embedding,
            )
        return self.norm_out(hidden_states, time_embedding)


class TrajectoryDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = NextDiT()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(hidden_states, timestep, encoder_hidden_states)


class System1Denoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_encoder = nn.Linear(3, 384)
        self.position_encoding = SinusoidalPositionEncoding(384)
        self.traj_dit = TrajectoryDiT()
        self.action_decoder = nn.Linear(384, 3)

    def forward(
        self,
        trajectories: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(trajectories.shape[1], device=trajectories.device).unsqueeze(0)
        encoded = self.action_encoder(trajectories) + self.position_encoding(positions)
        return self.action_decoder(self.traj_dit(encoded, timestep, context))


def load_system1_context_encoder(model_dir: Path) -> System1ContextEncoder:
    model = System1ContextEncoder()
    load_module_prefix(
        model.rgb_model,
        model_dir,
        "model.rgb_model.",
        ignored_checkpoint_keys=("mask_token",),
    )
    load_module_prefix(model.memory_encoder, model_dir, "model.memory_encoder.")
    load_module_prefix(model.rgb_resampler, model_dir, "model.rgb_resampler.")
    load_module_prefix(model.cond_projector, model_dir, "model.cond_projector.")
    return model.eval()


def load_system1_denoiser(model_dir: Path) -> System1Denoiser:
    model = System1Denoiser()
    load_module_prefix(model.action_encoder, model_dir, "model.action_encoder.")
    load_module_prefix(model.traj_dit, model_dir, "model.traj_dit.")
    load_module_prefix(model.action_decoder, model_dir, "model.action_decoder.")
    return model.eval()
